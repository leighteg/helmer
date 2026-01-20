use crate::graphics::renderer_common::{
    common::{
        Aabb, AssetStreamKind, AssetStreamingRequest, MeshLodPayload, MeshletLodData,
        RenderMessage, Vertex, build_mip_uploads, calc_mip_level_count, mip_level_data_size,
        render_message_payload_bytes,
    },
    meshlets::{build_meshlet_lod, meshlet_lod_size_bytes},
};
use crate::runtime::runtime::RuntimeTuning;
use base64::Engine;
use basis_universal::encoding::{ColorSpace, Compressor, CompressorParams};
use basis_universal::transcoding::{TranscodeParameters, Transcoder, TranscoderTextureFormat};
use basis_universal::{BasisTextureFormat, UASTC_QUALITY_MIN};
use crossbeam_channel::{Sender, TryRecvError, TrySendError, bounded};
use glam::{Mat4, Vec3};
use gltf::Texture as GltfTexture;
use hashbrown::{HashMap, HashSet};
use ktx2::Reader;
use memmap2::MmapOptions;
#[cfg(unix)]
use memmap2::UncheckedAdvice;
use meshopt::{SimplifyOptions, VertexDataAdapter, optimize_vertex_cache, simplify};
use parking_lot::RwLock;
use serde::Deserialize;
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::VecDeque,
    fs::File,
    marker::PhantomData,
    ops::Deref,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use tracing::{info, warn};

const FORWARD_TA_TARGET_RES: u32 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetKind {
    Albedo,
    Normal,
    MetallicRoughness,
    Emission,
}

// --- ASSET STRUCTS & HANDLES ---

#[derive(Debug, PartialEq, Eq, Hash)]
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

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self::new(self.id)
    }
}

impl<T> Copy for Handle<T> {}

fn read_image_bytes_from_source<'a, B: BufferSource>(
    source: &'a gltf::image::Source<'a>,
    buffers: &'a [B],
    base_path: Option<&'a Path>,
) -> Result<Option<(Cow<'a, [u8]>, Option<String>, Option<String>)>, String> {
    match source {
        gltf::image::Source::View { view, mime_type } => {
            let buffer = buffers.get(view.buffer().index()).ok_or_else(|| {
                format!("Missing buffer {} for image view", view.buffer().index())
            })?;
            let data = buffer.as_slice();
            let end = view.offset() + view.length();
            if end > data.len() {
                return Err(format!(
                    "Image view range {}..{} exceeds buffer length {}",
                    view.offset(),
                    end,
                    data.len()
                ));
            }
            Ok(Some((
                Cow::Borrowed(&data[view.offset()..end]),
                Some(mime_type.to_string()),
                None,
            )))
        }
        gltf::image::Source::Uri { uri, mime_type } => {
            let mut mime_hint = mime_type.map(|m| m.to_string());
            if let Some((bytes, uri_mime)) = decode_data_uri(uri)? {
                if mime_hint.is_none() {
                    mime_hint = uri_mime;
                }
                let ext_hint = uri.rsplit('.').next().map(|s| s.to_string());
                return Ok(Some((Cow::Owned(bytes), mime_hint, ext_hint)));
            }

            let base = match base_path {
                Some(b) => b,
                None => {
                    warn!(
                        "Skipping external image URI '{}' with no base path available",
                        uri
                    );
                    return Ok(None);
                }
            };
            let img_path = base.join(uri);
            match std::fs::read(&img_path) {
                Ok(bytes) => {
                    let ext_hint = img_path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.to_string());
                    Ok(Some((Cow::Owned(bytes), mime_hint, ext_hint)))
                }
                Err(e) => Err(format!(
                    "Failed to read external image '{}': {}",
                    img_path.display(),
                    e
                )),
            }
        }
    }
}

fn decode_texture_bytes(
    kind: AssetKind,
    bytes: &[u8],
    mime_hint: Option<&str>,
    ext_hint: Option<&str>,
) -> Result<(Vec<u8>, wgpu::TextureFormat, (u32, u32)), String> {
    let hint = mime_hint
        .map(|m| m.to_ascii_lowercase())
        .or_else(|| ext_hint.map(|e| e.to_ascii_lowercase()));

    let looks_like_ktx2 =
        matches!(hint.as_deref(), Some("image/ktx2") | Some("ktx2")) || is_ktx2_bytes(bytes);
    if looks_like_ktx2 {
        return decode_ktx2(bytes, kind);
    }

    let decoded = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
    let mut rgba = decoded.to_rgba8();
    let mut dimensions = rgba.dimensions();

    if std::env::var("HELMER_PATH") == Ok("forwardTA".to_string()) {
        if dimensions.0 != FORWARD_TA_TARGET_RES || dimensions.1 != FORWARD_TA_TARGET_RES {
            rgba = image::imageops::resize(
                &rgba,
                FORWARD_TA_TARGET_RES,
                FORWARD_TA_TARGET_RES,
                image::imageops::FilterType::Lanczos3,
            );
            dimensions = (FORWARD_TA_TARGET_RES, FORWARD_TA_TARGET_RES);
        }
    }

    let data = rgba.into_raw();
    let wgpu_format = match kind {
        AssetKind::Albedo | AssetKind::Emission => wgpu::TextureFormat::Rgba8UnormSrgb,
        _ => wgpu::TextureFormat::Rgba8Unorm,
    };
    Ok((data, wgpu_format, dimensions))
}

fn decode_texture_asset(
    gltf_texture: gltf::Texture,
    buffers: &[StreamedBuffer],
    base_path: Option<&Path>,
    kind: AssetKind,
) -> Option<(String, AssetKind, Vec<u8>, wgpu::TextureFormat, (u32, u32))> {
    let image = gltf_texture.source();
    let name = gltf_texture
        .name()
        .unwrap_or("unnamed_gltf_texture")
        .to_string();

    let image_source = image.source();
    let (pixels, mime_hint, ext_hint) =
        match read_image_bytes_from_source(&image_source, buffers, base_path) {
            Ok(Some(data)) => data,
            Ok(None) => return None,
            Err(e) => {
                warn!(
                    "Failed to process texture '{}': {}. Using fallback.",
                    name, e
                );
                return None;
            }
        };

    match decode_texture_bytes(
        kind,
        pixels.as_ref(),
        mime_hint.as_deref(),
        ext_hint.as_deref(),
    ) {
        Ok((data, format, dimensions)) => Some((name, kind, data, format, dimensions)),
        Err(e) => {
            warn!(
                "Failed to decode texture '{}': {}. Using fallback.",
                name, e
            );
            None
        }
    }
}

fn is_ktx2_bytes(bytes: &[u8]) -> bool {
    const KTX2_MAGIC: &[u8] = b"\xABKTX 20\xBB\r\n\x1A\n";
    bytes.len() >= KTX2_MAGIC.len() && &bytes[..KTX2_MAGIC.len()] == KTX2_MAGIC
}

fn decode_data_uri(uri: &str) -> Result<Option<(Vec<u8>, Option<String>)>, String> {
    if !uri.starts_with("data:") {
        return Ok(None);
    }
    let payload = &uri[5..];
    let mut parts = payload.splitn(2, ',');
    let meta = parts.next().unwrap_or_default();
    let data_part = parts
        .next()
        .ok_or_else(|| "Malformed data URI: missing data section".to_string())?;

    let (meta, is_base64) = if let Some(stripped) = meta.strip_suffix(";base64") {
        (stripped, true)
    } else {
        (meta, false)
    };

    if !is_base64 {
        return Err("Only base64-encoded data URIs are supported".to_string());
    }

    let decoded = base64::engine::general_purpose::STANDARD
        .decode(data_part)
        .map_err(|e| format!("Failed to decode data URI: {}", e))?;
    let mime_hint = if meta.is_empty() {
        None
    } else {
        Some(meta.to_string())
    };
    Ok(Some((decoded, mime_hint)))
}

#[derive(Debug)]
pub struct Mesh {
    pub lods: RwLock<Vec<MeshLodPayload>>,
    pub bounds: Aabb,
    pub total_lods: usize,
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

#[derive(Debug, Clone)]
struct CachedTexture {
    pub name: String,
    pub kind: AssetKind,
    pub data: Arc<[u8]>,
    pub format: wgpu::TextureFormat,
    pub dimensions: (u32, u32),
    pub low_res: Arc<RwLock<Option<LowResTexture>>>,
}

#[derive(Debug, Clone)]
struct LowResTexture {
    pub data: Arc<[u8]>,
    pub format: wgpu::TextureFormat,
    pub dimensions: (u32, u32),
}

fn generate_low_res_from_parts(
    data: &[u8],
    format: wgpu::TextureFormat,
    dimensions: (u32, u32),
    max_dim: u32,
) -> LowResTexture {
    let max_dim = max_dim.max(1);
    let (width, height) = dimensions;
    let base_size = mip_level_data_size(format, width, height);

    let mip_levels = calc_mip_level_count(width, height);
    let (layouts, _) = build_mip_uploads(format, width, height, data.len(), mip_levels);
    if let Some(layout) = layouts
        .iter()
        .rev()
        .find(|layout| layout.width <= max_dim && layout.height <= max_dim)
    {
        let end = layout.offset.saturating_add(layout.size);
        if end <= data.len() && layout.size > 0 {
            return LowResTexture {
                data: Arc::from(data[layout.offset..end].to_vec()),
                format,
                dimensions: (layout.width, layout.height),
            };
        }
    }

    let block_size = format.block_copy_size(None).unwrap_or(4) as usize;

    if block_size == 4 {
        if base_size <= data.len() {
            let base = &data[..base_size];
            if let Some(img) =
                image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, base.to_vec())
            {
                let target_w = width.min(max_dim).max(1);
                let target_h = height.min(max_dim).max(1);
                let resized = if width > max_dim || height > max_dim {
                    image::imageops::resize(
                        &img,
                        target_w,
                        target_h,
                        image::imageops::FilterType::Triangle,
                    )
                } else {
                    img
                };
                return LowResTexture {
                    data: Arc::from(resized.into_raw()),
                    format,
                    dimensions: (target_w, target_h),
                };
            }
        }
    }

    let (width, height) = if block_size > 4 { (4, 4) } else { (1, 1) };
    LowResTexture {
        data: Arc::from(vec![0; block_size.max(1)]),
        format,
        dimensions: (width, height),
    }
}

/// Temporary data returned by a worker thread after parsing a glTF file.
/// All indices here are local to this structure's vectors.
#[derive(Debug)]
struct ParsedGltfScene {
    doc: Arc<gltf::Document>,
    buffers: Option<Arc<Vec<StreamedBuffer>>>,
    buffers_bytes: usize,
    base_path: Option<PathBuf>,
    scene_path: Option<PathBuf>,
    textures: Vec<TextureRequest>,
    materials: Vec<IntermediateMaterial>,
    mesh_primitives: Vec<MeshPrimitiveDesc>,
    nodes: Vec<GltfNodeDesc>,
}

#[derive(Debug, Clone, Copy)]
struct MeshPrimitiveDesc {
    mesh_index: usize,
    primitive_index: usize,
    material_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct GltfNodeDesc {
    primitive_desc_index: usize,
    material_index: usize,
    transform: glam::Mat4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TextureRequest {
    tex_index: usize,
    kind: AssetKind,
}

/// An intermediate material representation that uses texture indices
/// instead of string paths, suitable for parsed scene data.
#[derive(Debug, Clone)]
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

trait BufferSource: Send + Sync {
    fn as_slice(&self) -> &[u8];
}

impl BufferSource for gltf::buffer::Data {
    fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

#[derive(Debug)]
struct StreamedBuffer {
    backing: StreamedBufferBacking,
}

#[derive(Debug)]
enum StreamedBufferBacking {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl StreamedBuffer {
    fn owned(data: Vec<u8>) -> Self {
        Self {
            backing: StreamedBufferBacking::Owned(data),
        }
    }

    fn mapped(map: memmap2::Mmap) -> Self {
        Self {
            backing: StreamedBufferBacking::Mapped(map),
        }
    }

    fn len(&self) -> usize {
        self.deref().len()
    }

    fn release_pages(&self) {
        #[cfg(unix)]
        {
            if let StreamedBufferBacking::Mapped(map) = &self.backing {
                // SAFETY: call only after streaming use; no further reads from this mapping.
                unsafe {
                    let _ = map.unchecked_advise(UncheckedAdvice::DontNeed);
                }
            }
        }
    }
}

impl Deref for StreamedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match &self.backing {
            StreamedBufferBacking::Owned(data) => data.as_slice(),
            StreamedBufferBacking::Mapped(map) => map.as_ref(),
        }
    }
}

impl BufferSource for StreamedBuffer {
    fn as_slice(&self) -> &[u8] {
        self.deref()
    }
}

struct SceneBufferGuard {
    buffers: Arc<Vec<StreamedBuffer>>,
    allow_release: bool,
}

impl SceneBufferGuard {
    fn new(buffers: Arc<Vec<StreamedBuffer>>, allow_release: bool) -> Self {
        Self {
            buffers,
            allow_release,
        }
    }

    fn buffers(&self) -> &[StreamedBuffer] {
        self.buffers.as_ref()
    }
}

impl Drop for SceneBufferGuard {
    fn drop(&mut self) {
        if !self.allow_release {
            return;
        }
        for buffer in self.buffers.iter() {
            buffer.release_pages();
        }
    }
}

enum SceneContextStatus {
    Ready {
        doc: Arc<gltf::Document>,
        buffers: SceneBufferGuard,
        base_path: Option<PathBuf>,
    },
    Pending,
    Missing,
}

// --- WORKER THREAD COMMUNICATION ---

enum AssetLoadRequest {
    Mesh {
        id: usize,
        path: PathBuf,
        tuning: AssetStreamingTuning,
    },
    ProceduralMesh {
        id: usize,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        tuning: AssetStreamingTuning,
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
    SceneBuffers {
        scene_id: usize,
        doc: Arc<gltf::Document>,
        base_path: Option<PathBuf>,
        scene_path: Option<PathBuf>,
    },
    StreamMesh {
        id: usize,
        scene_id: usize,
        desc: MeshPrimitiveDesc,
        doc: Arc<gltf::Document>,
        buffers: SceneBufferGuard,
        tuning: AssetStreamingTuning,
    },
    StreamTexture {
        id: usize,
        scene_id: usize,
        tex_index: usize,
        kind: AssetKind,
        doc: Arc<gltf::Document>,
        buffers: SceneBufferGuard,
        base_path: Option<PathBuf>,
    },
    LowResTexture {
        id: usize,
        name: String,
        kind: AssetKind,
        data: Arc<[u8]>,
        format: wgpu::TextureFormat,
        dimensions: (u32, u32),
        max_dim: u32,
    },
}

enum AssetLoadResult {
    Mesh {
        id: usize,
        scene_id: Option<usize>,
        lods: Vec<MeshLodPayload>,
        total_lods: usize,
        bounds: Aabb,
    },
    Texture {
        id: usize,
        scene_id: Option<usize>,
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
    SceneBuffers {
        scene_id: usize,
        buffers: Vec<StreamedBuffer>,
        buffers_bytes: usize,
    },
    SceneBuffersFailed {
        scene_id: usize,
    },
    LowResTexture {
        id: usize,
        name: String,
        kind: AssetKind,
        data: LowResTexture,
    },
    StreamFailure {
        kind: AssetStreamKind,
        id: usize,
        scene_id: usize,
    },
}

#[derive(Default)]
pub struct MeshAabbMap(pub HashMap<usize, Aabb>);

#[derive(Clone, Copy, Debug)]
pub struct AssetStreamingTuning {
    pub low_res_max_dim: u32,
    pub lod_safe_vertex_limit: usize,
    pub lod_safe_index_limit: usize,
    pub lod_targets: [f32; 3],
    pub lod_simplification_error: f32,
}

impl Default for AssetStreamingTuning {
    fn default() -> Self {
        Self {
            low_res_max_dim: 64,
            lod_safe_vertex_limit: 5_000_000,
            lod_safe_index_limit: 15_000_000,
            lod_targets: [0.7, 0.4, 0.15],
            lod_simplification_error: 0.02,
        }
    }
}

#[derive(Clone)]
struct SceneAssetContext {
    doc: Arc<gltf::Document>,
    buffers: Option<Arc<Vec<StreamedBuffer>>>,
    buffers_bytes: usize,
    buffers_last_used: Instant,
    base_path: Option<PathBuf>,
    scene_path: Option<PathBuf>,
    pending_assets: usize,
}

#[derive(Clone, Copy)]
struct TextureSource {
    scene_id: usize,
    request: TextureRequest,
}

#[derive(Clone, Copy)]
struct MeshSource {
    scene_id: usize,
    desc: MeshPrimitiveDesc,
}

#[derive(Clone)]
struct MaterialSource {
    scene_id: usize,
    data: MaterialGpuData,
}

// --- ASSET SERVER ---

pub struct AssetServer {
    scenes: Arc<RwLock<HashMap<usize, Arc<Scene>>>>,
    pub mesh_aabb_map: Arc<RwLock<MeshAabbMap>>,
    next_id: AtomicUsize,
    request_sender: crossbeam_channel::Sender<AssetLoadRequest>,
    result_receiver: crossbeam_channel::Receiver<AssetLoadResult>,
    asset_sender: Sender<RenderMessage>,
    tuning: Arc<RuntimeTuning>,
    _worker_handles: Vec<JoinHandle<()>>,
    pending_worker_requests: RwLock<VecDeque<AssetLoadRequest>>,

    stream_request_receiver: crossbeam_channel::Receiver<AssetStreamingRequest>,
    streaming_backlog: RwLock<VecDeque<AssetStreamingRequest>>,
    reupload_queue: RwLock<VecDeque<AssetStreamingRequest>>,
    streaming_inflight: RwLock<HashMap<(AssetStreamKind, usize), AssetStreamingRequest>>,
    latest_streaming_plan: RwLock<HashMap<(AssetStreamKind, usize), f32>>,
    scene_contexts: RwLock<HashMap<usize, SceneAssetContext>>,
    scene_buffer_bytes: AtomicUsize,
    scene_buffer_inflight: RwLock<HashSet<usize>>,
    texture_sources: RwLock<HashMap<usize, TextureSource>>,
    mesh_sources: RwLock<HashMap<usize, MeshSource>>,
    material_sources: RwLock<HashMap<usize, MaterialSource>>,
    mesh_cache: RwLock<HashMap<usize, Arc<Mesh>>>,
    texture_cache: RwLock<HashMap<usize, Arc<CachedTexture>>>,
    material_cache: RwLock<HashMap<usize, Arc<MaterialGpuData>>>,
    mesh_meta: RwLock<HashMap<usize, CacheEntryMeta>>,
    texture_meta: RwLock<HashMap<usize, CacheEntryMeta>>,
    material_meta: RwLock<HashMap<usize, CacheEntryMeta>>,
    mesh_budget: usize,
    texture_budget: usize,
    material_budget: usize,
    scene_buffer_budget: usize,
    asset_streaming_tuning: AssetStreamingTuning,

    // limit for how many assets to decode/create per frame to avoid stutter
    asset_creation_limit_per_frame: usize,
    streaming_upload_limit_per_frame: usize,
    streaming_backlog_limit: usize,
    cache_idle_ms: u64,
    cache_eviction_limit: usize,
    worker_queue_capacity: usize,
    low_res_inflight: RwLock<HashSet<usize>>,
}

#[derive(Debug, Clone)]
struct CacheEntryMeta {
    last_used: Instant,
    size_bytes: usize,
    priority: f32,
}

enum AssetSendError {
    Budget,
    Full,
    Disconnected,
}

impl AssetServer {
    pub fn new(
        asset_sender: Sender<RenderMessage>,
        stream_request_receiver: crossbeam_channel::Receiver<AssetStreamingRequest>,
        worker_queue_capacity: usize,
        tuning: Arc<RuntimeTuning>,
    ) -> Self {
        let (request_sender, request_receiver) = bounded(worker_queue_capacity);
        let (result_sender, result_receiver) = bounded(worker_queue_capacity);
        let mut worker_handles = Vec::new();

        let num_workers = num_cpus::get().min(4).max(1);
        for _ in 0..num_workers {
            let req_receiver = request_receiver.clone();
            let res_sender = result_sender.clone();
            let handle = thread::spawn(move || {
                while let Ok(request) = req_receiver.recv() {
                    let path_str = request_path(&request).to_string_lossy().to_string();
                    let result = match request {
                        AssetLoadRequest::Mesh { id, path, tuning } => {
                            let parsed = if is_gltf_path(&path) {
                                match parse_mesh_from_gltf_path(&path) {
                                    Ok(data) => Some(data),
                                    Err(e) => {
                                        warn!("Failed to parse glTF mesh '{}': {}", path_str, e);
                                        None
                                    }
                                }
                            } else {
                                match std::fs::read(&path) {
                                    Ok(bytes) => match parse_glb(&bytes) {
                                        Ok(data) => Some(data),
                                        Err(e) => {
                                            warn!("Failed to parse glb mesh '{}': {}", path_str, e);
                                            None
                                        }
                                    },
                                    Err(e) => {
                                        warn!(
                                            "Failed to read mesh file '{:?}' for handle {}: {}",
                                            path, id, e
                                        );
                                        None
                                    }
                                }
                            };

                            parsed.and_then(|(vertices, indices, bounds)| {
                                build_mesh_payload(vertices, indices, bounds, &tuning).map(
                                    |payload| AssetLoadResult::Mesh {
                                        id,
                                        scene_id: None,
                                        lods: payload.lods,
                                        total_lods: payload.total_lods,
                                        bounds: payload.bounds,
                                    },
                                )
                            })
                        }
                        AssetLoadRequest::ProceduralMesh {
                            id,
                            vertices,
                            indices,
                            tuning,
                        } => {
                            let bounds = Aabb::calculate(&vertices);
                            build_mesh_payload(vertices, indices, bounds, &tuning).map(|payload| {
                                AssetLoadResult::Mesh {
                                    id,
                                    scene_id: None,
                                    lods: payload.lods,
                                    total_lods: payload.total_lods,
                                    bounds: payload.bounds,
                                }
                            })
                        }
                        AssetLoadRequest::Texture { id, path, kind } => load_and_parse(
                            id,
                            &path,
                            |bytes| decode_ktx2(bytes, kind),
                            |id, data| AssetLoadResult::Texture {
                                id,
                                scene_id: None,
                                name: path_str.clone(),
                                kind,
                                data,
                            },
                        ),
                        AssetLoadRequest::Material { id, path } => {
                            load_and_parse(id, &path, parse_ron_material, |id, data| {
                                AssetLoadResult::Material { id, data }
                            })
                        }
                        AssetLoadRequest::Scene { id, path } => {
                            if is_gltf_path(&path) {
                                match parse_scene_from_gltf_path(&path) {
                                    Ok(data) => Some(AssetLoadResult::Scene { id, data }),
                                    Err(e) => {
                                        warn!("Failed to parse glTF scene '{}': {}", path_str, e);
                                        None
                                    }
                                }
                            } else {
                                match parse_scene_glb_path(&path) {
                                    Ok(data) => Some(AssetLoadResult::Scene { id, data }),
                                    Err(e) => {
                                        warn!("Failed to parse glTF scene '{}': {}", path_str, e);
                                        None
                                    }
                                }
                            }
                        }
                        AssetLoadRequest::SceneBuffers {
                            scene_id,
                            doc,
                            base_path,
                            scene_path,
                        } => match load_scene_buffers(
                            &doc,
                            scene_path.as_deref(),
                            base_path.as_deref(),
                        ) {
                            Ok(buffers) => {
                                let buffers_bytes: usize =
                                    buffers.iter().map(|buffer| buffer.len()).sum();
                                Some(AssetLoadResult::SceneBuffers {
                                    scene_id,
                                    buffers,
                                    buffers_bytes,
                                })
                            }
                            Err(err) => {
                                warn!("Failed to load buffers for scene {}: {}", scene_id, err);
                                Some(AssetLoadResult::SceneBuffersFailed { scene_id })
                            }
                        },
                        AssetLoadRequest::StreamMesh {
                            id,
                            scene_id,
                            desc,
                            doc,
                            buffers,
                            tuning,
                        } => {
                            let primitive = doc
                                .meshes()
                                .nth(desc.mesh_index)
                                .and_then(|mesh| mesh.primitives().nth(desc.primitive_index));
                            if let Some(primitive) = primitive {
                                if let Some((vertices, indices, bounds)) =
                                    process_primitive(&primitive, buffers.buffers())
                                {
                                    build_mesh_payload(vertices, indices, bounds, &tuning)
                                        .map_or_else(
                                            || {
                                                Some(AssetLoadResult::StreamFailure {
                                                    kind: AssetStreamKind::Mesh,
                                                    id,
                                                    scene_id,
                                                })
                                            },
                                            |payload| {
                                                Some(AssetLoadResult::Mesh {
                                                    id,
                                                    scene_id: Some(scene_id),
                                                    lods: payload.lods,
                                                    total_lods: payload.total_lods,
                                                    bounds: payload.bounds,
                                                })
                                            },
                                        )
                                } else {
                                    warn!(
                                        "Failed to process mesh primitive {} for streamed mesh {}",
                                        desc.primitive_index, id
                                    );
                                    Some(AssetLoadResult::StreamFailure {
                                        kind: AssetStreamKind::Mesh,
                                        id,
                                        scene_id,
                                    })
                                }
                            } else {
                                warn!(
                                    "Mesh {} primitive {} not found in scene {}",
                                    desc.mesh_index, desc.primitive_index, scene_id
                                );
                                Some(AssetLoadResult::StreamFailure {
                                    kind: AssetStreamKind::Mesh,
                                    id,
                                    scene_id,
                                })
                            }
                        }
                        AssetLoadRequest::StreamTexture {
                            id,
                            scene_id,
                            tex_index,
                            kind,
                            doc,
                            buffers,
                            base_path,
                        } => {
                            if let Some(gltf_tex) = doc.textures().nth(tex_index) {
                                let decoded = decode_texture_asset(
                                    gltf_tex,
                                    buffers.buffers(),
                                    base_path.as_deref(),
                                    kind,
                                );
                                decoded
                                    .map(|(name, kind, data, format, dimensions)| {
                                        AssetLoadResult::Texture {
                                            id,
                                            scene_id: Some(scene_id),
                                            name,
                                            kind,
                                            data: (data, format, dimensions),
                                        }
                                    })
                                    .or_else(|| {
                                        Some(AssetLoadResult::StreamFailure {
                                            kind: AssetStreamKind::Texture,
                                            id,
                                            scene_id,
                                        })
                                    })
                            } else {
                                warn!(
                                    "Texture {} index {} missing in scene {}",
                                    id, tex_index, scene_id
                                );
                                Some(AssetLoadResult::StreamFailure {
                                    kind: AssetStreamKind::Texture,
                                    id,
                                    scene_id,
                                })
                            }
                        }
                        AssetLoadRequest::LowResTexture {
                            id,
                            name,
                            kind,
                            data,
                            format,
                            dimensions,
                            max_dim,
                        } => {
                            let low = generate_low_res_from_parts(
                                data.as_ref(),
                                format,
                                dimensions,
                                max_dim,
                            );
                            Some(AssetLoadResult::LowResTexture {
                                id,
                                name,
                                kind,
                                data: low,
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

        let mb = 1024 * 1024;
        let mesh_budget = std::env::var("HELMER_ASSET_BUDGET_MESH_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256)
            * mb;
        let texture_budget = std::env::var("HELMER_ASSET_BUDGET_TEX_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024)
            * mb;
        let material_budget = std::env::var("HELMER_ASSET_BUDGET_MAT_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(128)
            * mb;
        let scene_buffer_budget = std::env::var("HELMER_ASSET_BUDGET_SCENE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256)
            * mb;
        let cache_idle_ms = std::env::var("HELMER_ASSET_CACHE_IDLE_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        let cache_eviction_limit = std::env::var("HELMER_ASSET_CACHE_EVICT_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);

        Self {
            scenes: Arc::new(RwLock::new(HashMap::new())),
            mesh_aabb_map: Arc::new(RwLock::new(MeshAabbMap::default())),
            next_id: AtomicUsize::new(0),
            request_sender,
            result_receiver,
            asset_sender,
            tuning,
            _worker_handles: worker_handles,
            pending_worker_requests: RwLock::new(VecDeque::new()),
            stream_request_receiver,
            streaming_backlog: RwLock::new(VecDeque::new()),
            reupload_queue: RwLock::new(VecDeque::new()),
            streaming_inflight: RwLock::new(HashMap::new()),
            latest_streaming_plan: RwLock::new(HashMap::new()),
            scene_contexts: RwLock::new(HashMap::new()),
            scene_buffer_bytes: AtomicUsize::new(0),
            scene_buffer_inflight: RwLock::new(HashSet::new()),
            texture_sources: RwLock::new(HashMap::new()),
            mesh_sources: RwLock::new(HashMap::new()),
            material_sources: RwLock::new(HashMap::new()),
            mesh_cache: RwLock::new(HashMap::new()),
            texture_cache: RwLock::new(HashMap::new()),
            material_cache: RwLock::new(HashMap::new()),
            mesh_meta: RwLock::new(HashMap::new()),
            texture_meta: RwLock::new(HashMap::new()),
            material_meta: RwLock::new(HashMap::new()),
            mesh_budget,
            texture_budget,
            material_budget,
            scene_buffer_budget,
            asset_streaming_tuning: AssetStreamingTuning::default(),
            asset_creation_limit_per_frame: 4,
            streaming_upload_limit_per_frame: 24,
            streaming_backlog_limit: 16_384,
            cache_idle_ms,
            cache_eviction_limit,
            worker_queue_capacity,
            low_res_inflight: RwLock::new(HashSet::new()),
        }
    }

    pub fn publish_streaming_plan(&self, plan: &[AssetStreamingRequest]) {
        let mut map = HashMap::with_capacity(plan.len());
        let mut material_hints: Vec<(usize, f32)> = Vec::new();
        for req in plan {
            map.entry((req.kind, req.id))
                .and_modify(|prio| {
                    if req.priority > *prio {
                        *prio = req.priority;
                    }
                })
                .or_insert(req.priority);
            if req.kind == AssetStreamKind::Material {
                material_hints.push((req.id, req.priority));
            }
        }

        for (mat_id, priority) in material_hints {
            if let Some(textures) = self.material_texture_ids(mat_id) {
                for tex in textures.iter().flatten() {
                    map.entry((AssetStreamKind::Texture, *tex))
                        .and_modify(|prio| {
                            if priority > *prio {
                                *prio = priority;
                            }
                        })
                        .or_insert(priority);
                }
            }
        }
        if !map.is_empty() {
            let now = Instant::now();
            let mut mesh_meta = self.mesh_meta.write();
            let mut material_meta = self.material_meta.write();
            let mut texture_meta = self.texture_meta.write();
            for ((kind, id), priority) in map.iter() {
                match kind {
                    AssetStreamKind::Mesh => {
                        if let Some(meta) = mesh_meta.get_mut(id) {
                            meta.last_used = now;
                            meta.priority = meta.priority.max(*priority);
                        }
                    }
                    AssetStreamKind::Material => {
                        if let Some(meta) = material_meta.get_mut(id) {
                            meta.last_used = now;
                            meta.priority = meta.priority.max(*priority);
                        }
                    }
                    AssetStreamKind::Texture => {
                        if let Some(meta) = texture_meta.get_mut(id) {
                            meta.last_used = now;
                            meta.priority = meta.priority.max(*priority);
                        }
                    }
                }
            }
        }
        *self.latest_streaming_plan.write() = map;
    }

    fn stream_kind_rank(kind: AssetStreamKind) -> u8 {
        match kind {
            AssetStreamKind::Mesh => 0,
            AssetStreamKind::Material => 1,
            AssetStreamKind::Texture => 2,
        }
    }

    fn plan_priority(&self, kind: AssetStreamKind, id: usize) -> f32 {
        self.latest_streaming_plan
            .read()
            .get(&(kind, id))
            .copied()
            .unwrap_or(0.0)
    }

    fn material_texture_ids(&self, material_id: usize) -> Option<[Option<usize>; 4]> {
        if let Some(mat) = self.material_cache.read().get(&material_id) {
            return Some([
                mat.albedo_texture_id,
                mat.normal_texture_id,
                mat.metallic_roughness_texture_id,
                mat.emission_texture_id,
            ]);
        }
        if let Some(mat) = self.material_sources.read().get(&material_id) {
            let data = &mat.data;
            return Some([
                data.albedo_texture_id,
                data.normal_texture_id,
                data.metallic_roughness_texture_id,
                data.emission_texture_id,
            ]);
        }
        None
    }

    fn cache_meta_maps(
        &self,
        kind: AssetStreamKind,
    ) -> (&RwLock<HashMap<usize, CacheEntryMeta>>, usize) {
        match kind {
            AssetStreamKind::Mesh => (&self.mesh_meta, self.mesh_budget),
            AssetStreamKind::Material => (&self.material_meta, self.material_budget),
            AssetStreamKind::Texture => (&self.texture_meta, self.texture_budget),
        }
    }

    fn has_stream_source(&self, kind: AssetStreamKind, id: usize) -> bool {
        match kind {
            AssetStreamKind::Mesh => self.mesh_sources.read().contains_key(&id),
            AssetStreamKind::Material => self.material_sources.read().contains_key(&id),
            AssetStreamKind::Texture => self.texture_sources.read().contains_key(&id),
        }
    }

    fn record_cache_entry(&self, kind: AssetStreamKind, id: usize, size_bytes: usize) {
        let priority = self.plan_priority(kind, id);
        let (meta_map, _) = self.cache_meta_maps(kind);
        meta_map.write().insert(
            id,
            CacheEntryMeta {
                last_used: Instant::now(),
                size_bytes,
                priority,
            },
        );
    }

    fn update_cache_size(&self, kind: AssetStreamKind, id: usize, size_bytes: usize) {
        let priority = self.plan_priority(kind, id);
        let (meta_map, _) = self.cache_meta_maps(kind);
        let mut meta = meta_map.write();
        meta.entry(id)
            .and_modify(|m| {
                m.size_bytes = size_bytes;
                m.last_used = Instant::now();
                m.priority = m.priority.max(priority);
            })
            .or_insert(CacheEntryMeta {
                last_used: Instant::now(),
                size_bytes,
                priority,
            });
    }

    fn touch_cache_entry(&self, kind: AssetStreamKind, id: usize) {
        let priority = self.plan_priority(kind, id);
        let (meta_map, _) = self.cache_meta_maps(kind);
        if let Some(meta) = meta_map.write().get_mut(&id) {
            meta.last_used = Instant::now();
            meta.priority = meta.priority.max(priority);
        }
    }

    fn enforce_cache_budget(&self, kind: AssetStreamKind) {
        let (meta_map, budget) = self.cache_meta_maps(kind);
        let mut meta = meta_map.write();
        let mut total: usize = meta.values().map(|m| m.size_bytes).sum();
        if total <= budget {
            return;
        }
        let mut entries: Vec<(f32, Instant, usize, usize)> = meta
            .iter()
            .map(|(id, m)| (m.priority, m.last_used, *id, m.size_bytes))
            .collect();
        entries.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        for (_prio, _last, id, size) in entries {
            if total <= budget {
                break;
            }
            if !self.has_stream_source(kind, id) {
                continue;
            }
            meta.remove(&id);
            match kind {
                AssetStreamKind::Mesh => {
                    self.mesh_cache.write().remove(&id);
                }
                AssetStreamKind::Material => {
                    self.material_cache.write().remove(&id);
                }
                AssetStreamKind::Texture => {
                    self.texture_cache.write().remove(&id);
                }
            }
            total = total.saturating_sub(size);
        }
    }

    fn enforce_scene_buffer_budget(&self) {
        let budget = self.scene_buffer_budget;
        let mut total = self.scene_buffer_bytes.load(AtomicOrdering::Relaxed);
        if total == 0 {
            return;
        }

        let active_scenes = if budget == 0 || total > budget {
            self.active_streaming_scene_ids()
        } else {
            HashSet::new()
        };

        let mut contexts = self.scene_contexts.write();

        if budget == 0 {
            for (scene_id, ctx) in contexts.iter_mut() {
                if ctx.pending_assets > 0 {
                    continue;
                }
                if active_scenes.contains(scene_id) {
                    continue;
                }
                if let Some(buffers) = ctx.buffers.take() {
                    self.release_scene_buffers_if_needed(buffers.as_ref());
                    total = total.saturating_sub(ctx.buffers_bytes);
                    ctx.buffers_bytes = 0;
                }
            }
            self.scene_buffer_bytes
                .store(total, AtomicOrdering::Relaxed);
            return;
        }

        if total <= budget {
            return;
        }

        let mut entries: Vec<(Instant, usize, usize)> = contexts
            .iter()
            .filter_map(|(scene_id, ctx)| {
                if ctx.pending_assets > 0 {
                    return None;
                }
                if active_scenes.contains(scene_id) {
                    return None;
                }
                ctx.buffers
                    .as_ref()
                    .map(|_| (ctx.buffers_last_used, *scene_id, ctx.buffers_bytes))
            })
            .collect();
        if entries.is_empty() {
            return;
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        for (_last_used, scene_id, bytes) in entries {
            if total <= budget {
                break;
            }
            if let Some(ctx) = contexts.get_mut(&scene_id) {
                if ctx.buffers.take().is_some() {
                    total = total.saturating_sub(bytes);
                    ctx.buffers_bytes = 0;
                }
            }
        }
        self.scene_buffer_bytes
            .store(total, AtomicOrdering::Relaxed);
    }

    fn sync_mesh_cache_size(&self, id: usize, mesh: &Mesh) {
        let size_bytes = mesh_size_bytes(mesh);
        self.update_cache_size(AssetStreamKind::Mesh, id, size_bytes);
        self.enforce_cache_budget(AssetStreamKind::Mesh);
    }

    fn collect_idle_cache_ids(
        &self,
        kind: AssetStreamKind,
        meta_map: &RwLock<HashMap<usize, CacheEntryMeta>>,
        plan: &HashMap<(AssetStreamKind, usize), f32>,
        idle: Duration,
        now: Instant,
        limit: usize,
    ) -> Vec<usize> {
        if limit == 0 {
            return Vec::new();
        }

        let mut ids = Vec::new();
        for (id, meta) in meta_map.read().iter() {
            if ids.len() >= limit {
                break;
            }
            if plan.contains_key(&(kind, *id)) {
                continue;
            }
            if now.duration_since(meta.last_used) < idle {
                continue;
            }
            if !self.has_stream_source(kind, *id) {
                continue;
            }
            ids.push(*id);
        }
        ids
    }

    fn evict_idle_cache_entries(&self) {
        let idle_ms = self.cache_idle_ms;
        let limit = self.cache_eviction_limit;
        if idle_ms == 0 || limit == 0 {
            return;
        }

        let idle = Duration::from_millis(idle_ms);
        let now = Instant::now();
        let plan = self.latest_streaming_plan.read();
        let mut remaining = limit;

        let mesh_ids = self.collect_idle_cache_ids(
            AssetStreamKind::Mesh,
            &self.mesh_meta,
            &plan,
            idle,
            now,
            remaining,
        );
        if !mesh_ids.is_empty() {
            let mut cache = self.mesh_cache.write();
            let mut meta = self.mesh_meta.write();
            for id in mesh_ids.iter() {
                cache.remove(id);
                meta.remove(id);
            }
            remaining = remaining.saturating_sub(mesh_ids.len());
        }

        if remaining == 0 {
            return;
        }

        let material_ids = self.collect_idle_cache_ids(
            AssetStreamKind::Material,
            &self.material_meta,
            &plan,
            idle,
            now,
            remaining,
        );
        if !material_ids.is_empty() {
            let mut cache = self.material_cache.write();
            let mut meta = self.material_meta.write();
            for id in material_ids.iter() {
                cache.remove(id);
                meta.remove(id);
            }
            remaining = remaining.saturating_sub(material_ids.len());
        }

        if remaining == 0 {
            return;
        }

        let texture_ids = self.collect_idle_cache_ids(
            AssetStreamKind::Texture,
            &self.texture_meta,
            &plan,
            idle,
            now,
            remaining,
        );
        if !texture_ids.is_empty() {
            let mut cache = self.texture_cache.write();
            let mut meta = self.texture_meta.write();
            for id in texture_ids.iter() {
                cache.remove(id);
                meta.remove(id);
            }
        }
    }

    pub fn add_mesh(&self, vertices: Vec<Vertex>, indices: Vec<u32>) -> Handle<Mesh> {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);

        let request = AssetLoadRequest::ProceduralMesh {
            id,
            vertices,
            indices,
            tuning: self.asset_streaming_tuning,
        };

        if !self.enqueue_worker_request(request) {
            warn!("Failed to queue procedural mesh request; worker thread offline");
        }

        Handle::new(id)
    }

    pub fn set_limits(
        &mut self,
        asset_creation_limit_per_frame: usize,
        streaming_upload_limit_per_frame: usize,
    ) {
        self.asset_creation_limit_per_frame = asset_creation_limit_per_frame;
        self.streaming_upload_limit_per_frame = streaming_upload_limit_per_frame;
    }

    pub fn limits(&self) -> (usize, usize) {
        (
            self.asset_creation_limit_per_frame,
            self.streaming_upload_limit_per_frame,
        )
    }

    pub fn streaming_backlog_limit(&self) -> usize {
        self.streaming_backlog_limit
    }

    pub fn set_streaming_backlog_limit(&mut self, limit: usize) {
        self.streaming_backlog_limit = limit;
    }

    pub fn cache_idle_ms(&self) -> u64 {
        self.cache_idle_ms
    }

    pub fn set_cache_idle_ms(&mut self, idle_ms: u64) {
        self.cache_idle_ms = idle_ms;
    }

    pub fn cache_eviction_limit(&self) -> usize {
        self.cache_eviction_limit
    }

    pub fn set_cache_eviction_limit(&mut self, limit: usize) {
        self.cache_eviction_limit = limit;
    }

    pub fn asset_streaming_tuning(&self) -> AssetStreamingTuning {
        self.asset_streaming_tuning
    }

    pub fn set_asset_streaming_tuning(&mut self, tuning: AssetStreamingTuning) {
        let previous = self.asset_streaming_tuning;
        self.asset_streaming_tuning = tuning;

        if tuning.low_res_max_dim != previous.low_res_max_dim {
            self.invalidate_low_res_textures();
        }

        let lod_changed = tuning.lod_safe_vertex_limit != previous.lod_safe_vertex_limit
            || tuning.lod_safe_index_limit != previous.lod_safe_index_limit
            || tuning.lod_targets != previous.lod_targets
            || (tuning.lod_simplification_error - previous.lod_simplification_error).abs()
                > f32::EPSILON;
        if lod_changed {
            self.invalidate_mesh_lods();
        }
    }

    pub fn set_budget_bytes(
        &mut self,
        mesh_bytes: usize,
        texture_bytes: usize,
        material_bytes: usize,
    ) {
        self.mesh_budget = mesh_bytes;
        self.texture_budget = texture_bytes;
        self.material_budget = material_bytes;
        self.enforce_cache_budget(AssetStreamKind::Mesh);
        self.enforce_cache_budget(AssetStreamKind::Texture);
        self.enforce_cache_budget(AssetStreamKind::Material);
    }

    pub fn budgets_bytes(&self) -> (usize, usize, usize) {
        (self.mesh_budget, self.texture_budget, self.material_budget)
    }

    pub fn scene_buffer_budget_bytes(&self) -> usize {
        self.scene_buffer_budget
    }

    pub fn set_scene_buffer_budget_bytes(&mut self, budget: usize) {
        self.scene_buffer_budget = budget;
        self.enforce_scene_buffer_budget();
    }

    pub fn scene_buffer_usage_bytes(&self) -> usize {
        self.scene_buffer_bytes.load(AtomicOrdering::Relaxed)
    }

    pub fn cache_usage_bytes(&self) -> (usize, usize, usize) {
        let mesh = self.mesh_meta.read().values().map(|m| m.size_bytes).sum();
        let tex = self
            .texture_meta
            .read()
            .values()
            .map(|m| m.size_bytes)
            .sum();
        let mat = self
            .material_meta
            .read()
            .values()
            .map(|m| m.size_bytes)
            .sum();
        (mesh, tex, mat)
    }

    fn invalidate_low_res_textures(&self) {
        for tex in self.texture_cache.read().values() {
            *tex.low_res.write() = None;
        }
    }

    fn invalidate_mesh_lods(&self) {
        let meshes: Vec<(usize, Arc<Mesh>)> = self
            .mesh_cache
            .read()
            .iter()
            .map(|(id, mesh)| (*id, mesh.clone()))
            .collect();
        let mut mesh_meta = self.mesh_meta.write();
        for (id, mesh) in meshes {
            if let Some(meta) = mesh_meta.get_mut(&id) {
                meta.size_bytes = mesh_size_bytes(mesh.as_ref());
            }
        }
    }

    /// Resend all cached assets to the render thread. Useful after an explicit GPU eviction.
    pub fn reupload_cached_assets(&self) {
        let mesh_entries: Vec<(usize, Aabb, usize, f32)> = {
            let mesh_cache = self.mesh_cache.read();
            let mesh_meta = self.mesh_meta.read();
            mesh_cache
                .iter()
                .map(|(id, mesh)| {
                    let priority = mesh_meta.get(id).map(|m| m.priority).unwrap_or(0.0);
                    (*id, mesh.bounds, mesh_size_bytes(mesh.as_ref()), priority)
                })
                .collect()
        };
        let texture_entries: Vec<(usize, f32)> = {
            let texture_cache = self.texture_cache.read();
            let texture_meta = self.texture_meta.read();
            texture_cache
                .keys()
                .map(|id| {
                    let priority = texture_meta.get(id).map(|m| m.priority).unwrap_or(0.0);
                    (*id, priority)
                })
                .collect()
        };
        let material_entries: Vec<(usize, f32)> = {
            let material_cache = self.material_cache.read();
            let material_meta = self.material_meta.read();
            material_cache
                .keys()
                .map(|id| {
                    let priority = material_meta.get(id).map(|m| m.priority).unwrap_or(0.0);
                    (*id, priority)
                })
                .collect()
        };

        let mut requests: Vec<AssetStreamingRequest> =
            Vec::with_capacity(mesh_entries.len() + texture_entries.len() + material_entries.len());
        for (id, _bounds, _size, priority) in &mesh_entries {
            requests.push(AssetStreamingRequest {
                id: *id,
                kind: AssetStreamKind::Mesh,
                priority: *priority,
                max_lod: None,
                force_low_res: false,
            });
        }
        for (id, priority) in &texture_entries {
            requests.push(AssetStreamingRequest {
                id: *id,
                kind: AssetStreamKind::Texture,
                priority: *priority,
                max_lod: None,
                force_low_res: false,
            });
        }
        for (id, priority) in &material_entries {
            requests.push(AssetStreamingRequest {
                id: *id,
                kind: AssetStreamKind::Material,
                priority: *priority,
                max_lod: None,
                force_low_res: false,
            });
        }
        requests.sort_by(|a, b| {
            b.priority
                .total_cmp(&a.priority)
                .then_with(|| Self::stream_kind_rank(a.kind).cmp(&Self::stream_kind_rank(b.kind)))
                .then_with(|| a.id.cmp(&b.id))
        });

        {
            let mut queue = self.reupload_queue.write();
            queue.clear();
            queue.extend(requests);
        }

        if !mesh_entries.is_empty() {
            let mut map = self.mesh_aabb_map.write();
            for (id, bounds, _size, _priority) in &mesh_entries {
                map.0.insert(*id, *bounds);
            }
        }
        for (id, _bounds, size, _priority) in &mesh_entries {
            self.update_cache_size(AssetStreamKind::Mesh, *id, *size);
            self.touch_cache_entry(AssetStreamKind::Mesh, *id);
        }
        if !mesh_entries.is_empty() {
            self.enforce_cache_budget(AssetStreamKind::Mesh);
        }
        for (id, _priority) in &texture_entries {
            self.touch_cache_entry(AssetStreamKind::Texture, *id);
        }
        for (id, _priority) in &material_entries {
            self.touch_cache_entry(AssetStreamKind::Material, *id);
        }
    }

    fn try_send_asset_message(&self, message: RenderMessage) -> Result<(), AssetSendError> {
        let bytes = render_message_payload_bytes(&message);
        if bytes > 0 && !self.tuning.try_reserve_asset_upload(bytes) {
            return Err(AssetSendError::Budget);
        }

        match self.asset_sender.try_send(message) {
            Ok(_) => Ok(()),
            Err(TrySendError::Full(_)) => {
                if bytes > 0 {
                    self.tuning.release_asset_upload(bytes);
                }
                Err(AssetSendError::Full)
            }
            Err(TrySendError::Disconnected(_)) => {
                if bytes > 0 {
                    self.tuning.release_asset_upload(bytes);
                }
                Err(AssetSendError::Disconnected)
            }
        }
    }

    fn enqueue_worker_request(&self, request: AssetLoadRequest) -> bool {
        match self.request_sender.try_send(request) {
            Ok(()) => true,
            Err(TrySendError::Full(request)) => {
                self.pending_worker_requests.write().push_back(request);
                true
            }
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    fn flush_worker_requests(&self) {
        let mut pending = self.pending_worker_requests.write();
        if pending.is_empty() {
            return;
        }
        let mut budget = self.worker_queue_capacity.max(8);
        while budget > 0 {
            let Some(request) = pending.pop_front() else {
                break;
            };
            match self.request_sender.try_send(request) {
                Ok(()) => {
                    budget = budget.saturating_sub(1);
                }
                Err(TrySendError::Full(request)) => {
                    pending.push_front(request);
                    break;
                }
                Err(TrySendError::Disconnected(_)) => {
                    break;
                }
            }
        }
    }
    pub fn load_mesh<P: AsRef<Path>>(&self, path: P) -> Handle<Mesh> {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        let request = AssetLoadRequest::Mesh {
            id,
            path: path.as_ref().to_path_buf(),
            tuning: self.asset_streaming_tuning,
        };
        if !self.enqueue_worker_request(request) {
            warn!("Failed to queue mesh request {}; worker thread offline", id);
        }
        Handle {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn load_texture<P: AsRef<Path>>(&self, path: P, kind: AssetKind) -> Handle<Texture> {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        let request = AssetLoadRequest::Texture {
            id,
            path: path.as_ref().to_path_buf(),
            kind,
        };
        if !self.enqueue_worker_request(request) {
            warn!(
                "Failed to queue texture request {}; worker thread offline",
                id
            );
        }
        Handle {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn load_material<P: AsRef<Path>>(&self, path: P) -> Handle<Material> {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        let request = AssetLoadRequest::Material {
            id,
            path: path.as_ref().to_path_buf(),
        };
        if !self.enqueue_worker_request(request) {
            warn!(
                "Failed to queue material request {}; worker thread offline",
                id
            );
        }
        Handle {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn load_scene<P: AsRef<Path>>(&self, path: P) -> Handle<Scene> {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        let request = AssetLoadRequest::Scene {
            id,
            path: path.as_ref().to_path_buf(),
        };
        if !self.enqueue_worker_request(request) {
            warn!(
                "Failed to queue scene request {}; worker thread offline",
                id
            );
        }
        Handle {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn get_scene(&self, handle: &Handle<Scene>) -> Option<Arc<Scene>> {
        self.scenes.read().get(&handle.id).cloned()
    }

    pub fn update(&self) {
        self.flush_worker_requests();
        // --- 1. Drain all completed load results from workers ---
        let mut creation_budget = self.asset_creation_limit_per_frame;
        // Limit how many worker results we process per tick to avoid long stalls on huge loads.
        let mut results_budget = self.asset_creation_limit_per_frame.saturating_mul(4).max(4);
        while results_budget > 0 {
            match self.result_receiver.try_recv() {
                Ok(result) => {
                    results_budget = results_budget.saturating_sub(1);
                    match result {
                        AssetLoadResult::Mesh {
                            id,
                            scene_id,
                            lods,
                            total_lods,
                            bounds,
                        } => {
                            let mesh_arc = Arc::new(Mesh {
                                lods: RwLock::new(lods.clone()),
                                bounds,
                                total_lods,
                            });

                            let cached = self.mesh_budget > 0;
                            if cached {
                                let size_bytes = mesh_size_bytes(&mesh_arc);
                                self.mesh_cache.write().insert(id, mesh_arc.clone());
                                self.record_cache_entry(AssetStreamKind::Mesh, id, size_bytes);
                                self.enforce_cache_budget(AssetStreamKind::Mesh);
                            }

                            self.mesh_aabb_map.write().0.insert(id, bounds);
                            if let Some(scene_id) = scene_id {
                                self.mark_scene_asset_complete(scene_id);
                            }

                            let request = scene_id
                                .and_then(|_| {
                                    self.take_streaming_inflight(AssetStreamKind::Mesh, id)
                                })
                                .unwrap_or_else(|| AssetStreamingRequest {
                                    id,
                                    kind: AssetStreamKind::Mesh,
                                    priority: 0.0,
                                    max_lod: Some(0),
                                    force_low_res: false,
                                });

                            let lods_to_send = {
                                let lods = mesh_arc.lods.read();
                                select_mesh_lod_payloads(&lods, request.max_lod)
                            };
                            if lods_to_send.is_empty() {
                                continue;
                            }

                            match self.try_send_asset_message(RenderMessage::CreateMesh {
                                id,
                                total_lods: mesh_arc.total_lods,
                                lods: lods_to_send,
                                bounds,
                            }) {
                                Ok(()) => {
                                    if cached {
                                        self.touch_cache_entry(AssetStreamKind::Mesh, id);
                                    }
                                }
                                Err(AssetSendError::Budget) | Err(AssetSendError::Full) => {
                                    self.requeue_stream_request(request);
                                }
                                Err(AssetSendError::Disconnected) => {
                                    warn!("Failed to send mesh {}; render thread offline", id);
                                }
                            }
                        }
                        AssetLoadResult::Texture {
                            id,
                            scene_id,
                            name,
                            kind,
                            data,
                        } => {
                            let (texture_data, format, (width, height)) = data;
                            let texture_data: Arc<[u8]> = Arc::from(texture_data);
                            let cached = self.texture_budget > 0;
                            if cached {
                                let size_bytes = texture_data.len();
                                self.texture_cache.write().insert(
                                    id,
                                    Arc::new(CachedTexture {
                                        name: name.clone(),
                                        kind,
                                        data: texture_data.clone(),
                                        format,
                                        dimensions: (width, height),
                                        low_res: Arc::new(RwLock::new(None)),
                                    }),
                                );
                                self.record_cache_entry(AssetStreamKind::Texture, id, size_bytes);
                                self.enforce_cache_budget(AssetStreamKind::Texture);
                            }

                            if let Some(scene_id) = scene_id {
                                self.mark_scene_asset_complete(scene_id);
                            }

                            let request = scene_id
                                .and_then(|_| {
                                    self.take_streaming_inflight(AssetStreamKind::Texture, id)
                                })
                                .unwrap_or_else(|| AssetStreamingRequest {
                                    id,
                                    kind: AssetStreamKind::Texture,
                                    priority: 0.0,
                                    max_lod: None,
                                    force_low_res: false,
                                });

                            if request.force_low_res {
                                let cached_tex = if cached {
                                    self.texture_cache.read().get(&id).cloned()
                                } else {
                                    None
                                };
                                if let Some(tex) = cached_tex {
                                    if let Some(low) = tex.low_res.read().clone() {
                                        match self.try_send_asset_message(
                                            RenderMessage::CreateTexture {
                                                id,
                                                name: tex.name.clone(),
                                                kind: tex.kind,
                                                data: low.data.clone(),
                                                format: low.format,
                                                width: low.dimensions.0,
                                                height: low.dimensions.1,
                                            },
                                        ) {
                                            Ok(()) => {
                                                self.touch_cache_entry(
                                                    AssetStreamKind::Texture,
                                                    id,
                                                );
                                            }
                                            Err(AssetSendError::Budget)
                                            | Err(AssetSendError::Full) => {
                                                self.requeue_stream_request(request);
                                            }
                                            Err(AssetSendError::Disconnected) => {
                                                warn!(
                                                    "Failed to send low-res texture {}; render thread offline",
                                                    id
                                                );
                                            }
                                        }
                                    } else {
                                        self.queue_low_res_generation(
                                            id,
                                            tex.name.clone(),
                                            tex.kind,
                                            tex.data.clone(),
                                            tex.format,
                                            tex.dimensions,
                                        );
                                    }
                                } else {
                                    self.queue_low_res_generation(
                                        id,
                                        name.clone(),
                                        kind,
                                        texture_data.clone(),
                                        format,
                                        (width, height),
                                    );
                                }
                            } else {
                                match self.try_send_asset_message(RenderMessage::CreateTexture {
                                    id,
                                    name,
                                    kind,
                                    data: texture_data,
                                    format,
                                    width,
                                    height,
                                }) {
                                    Ok(()) => {
                                        if cached {
                                            self.touch_cache_entry(AssetStreamKind::Texture, id);
                                        }
                                    }
                                    Err(AssetSendError::Budget) | Err(AssetSendError::Full) => {
                                        self.requeue_stream_request(request);
                                    }
                                    Err(AssetSendError::Disconnected) => {
                                        warn!(
                                            "Failed to send texture {}; render thread offline",
                                            id
                                        );
                                    }
                                }
                            }
                        }
                        AssetLoadResult::Material { id, data } => {
                            let material_gpu_data = self.resolve_material_dependencies(id, data);
                            let size_bytes = std::mem::size_of_val(&material_gpu_data);
                            self.material_cache
                                .write()
                                .insert(id, Arc::new(material_gpu_data.clone()));
                            self.record_cache_entry(AssetStreamKind::Material, id, size_bytes);
                            self.enforce_cache_budget(AssetStreamKind::Material);
                            let request = AssetStreamingRequest {
                                id,
                                kind: AssetStreamKind::Material,
                                priority: 0.0,
                                max_lod: None,
                                force_low_res: false,
                            };
                            match self.try_send_asset_message(RenderMessage::CreateMaterial(
                                material_gpu_data,
                            )) {
                                Ok(()) => {
                                    self.touch_cache_entry(AssetStreamKind::Material, id);
                                }
                                Err(AssetSendError::Budget) | Err(AssetSendError::Full) => {
                                    self.requeue_stream_request(request);
                                }
                                Err(AssetSendError::Disconnected) => {
                                    warn!("Failed to send material {}; render thread offline", id);
                                }
                            }
                        }
                        AssetLoadResult::Scene { id: scene_id, data } => {
                            self.register_scene(scene_id, data);
                        }
                        AssetLoadResult::SceneBuffers {
                            scene_id,
                            buffers,
                            buffers_bytes,
                        } => {
                            self.scene_buffer_inflight.write().remove(&scene_id);
                            let stored = {
                                let mut contexts = self.scene_contexts.write();
                                let mut stored = false;
                                if let Some(ctx) = contexts.get_mut(&scene_id) {
                                    if ctx.buffers.is_none() {
                                        ctx.buffers = Some(Arc::new(buffers));
                                        ctx.buffers_bytes = buffers_bytes;
                                        stored = true;
                                    }
                                    ctx.buffers_last_used = Instant::now();
                                }
                                stored
                            };
                            if stored && buffers_bytes > 0 {
                                self.scene_buffer_bytes
                                    .fetch_add(buffers_bytes, AtomicOrdering::Relaxed);
                                self.enforce_scene_buffer_budget();
                            }
                        }
                        AssetLoadResult::SceneBuffersFailed { scene_id } => {
                            self.scene_buffer_inflight.write().remove(&scene_id);
                        }
                        AssetLoadResult::LowResTexture {
                            id,
                            name,
                            kind,
                            data,
                        } => {
                            self.low_res_inflight.write().remove(&id);
                            if let Some(tex) = self.texture_cache.read().get(&id) {
                                *tex.low_res.write() = Some(data.clone());
                                self.touch_cache_entry(AssetStreamKind::Texture, id);
                            }

                            let request = AssetStreamingRequest {
                                id,
                                kind: AssetStreamKind::Texture,
                                priority: 0.0,
                                max_lod: None,
                                force_low_res: true,
                            };
                            match self.try_send_asset_message(RenderMessage::CreateTexture {
                                id,
                                name,
                                kind,
                                data: data.data.clone(),
                                format: data.format,
                                width: data.dimensions.0,
                                height: data.dimensions.1,
                            }) {
                                Ok(()) => {}
                                Err(AssetSendError::Budget) | Err(AssetSendError::Full) => {
                                    self.requeue_stream_request(request);
                                }
                                Err(AssetSendError::Disconnected) => {
                                    warn!(
                                        "Failed to send low-res texture {}; render thread offline",
                                        id
                                    );
                                }
                            }
                        }
                        AssetLoadResult::StreamFailure { kind, id, scene_id } => {
                            self.take_streaming_inflight(kind, id);
                            self.mark_scene_asset_complete(scene_id);
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        // --- 2. Service any reupload requests before regular streaming ---
        let mut upload_budget = self.streaming_upload_limit_per_frame;
        if upload_budget > 0 {
            let reupload_sent = self.process_reupload_queue(upload_budget);
            upload_budget = upload_budget.saturating_sub(reupload_sent);
        }
        // --- 3. Service any streaming requests from the render thread ---
        self.process_stream_requests(upload_budget, &mut creation_budget);
        self.evict_idle_cache_entries();
    }

    fn register_scene(&self, scene_id: usize, data: ParsedGltfScene) {
        info!(
            "Scene {} parsed; registering layout for streaming-based loading.",
            scene_id
        );

        let ParsedGltfScene {
            doc,
            buffers,
            buffers_bytes,
            base_path,
            scene_path,
            textures,
            materials,
            mesh_primitives,
            nodes,
        } = data;

        let doc_clone = doc.clone();
        let context = SceneAssetContext {
            doc,
            buffers,
            buffers_bytes,
            buffers_last_used: Instant::now(),
            base_path: base_path.clone(),
            scene_path,
            pending_assets: textures.len() + materials.len() + mesh_primitives.len(),
        };
        self.scene_contexts.write().insert(scene_id, context);
        if buffers_bytes > 0 {
            self.scene_buffer_bytes
                .fetch_add(buffers_bytes, AtomicOrdering::Relaxed);
            self.enforce_scene_buffer_budget();
        }

        let mut texture_ids = Vec::with_capacity(textures.len());
        {
            let mut texture_sources = self.texture_sources.write();
            for req in textures.iter() {
                let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
                texture_ids.push(id);
                texture_sources.insert(
                    id,
                    TextureSource {
                        scene_id,
                        request: *req,
                    },
                );
            }
        }

        let mut material_ids = Vec::with_capacity(materials.len());
        {
            let mut material_sources = self.material_sources.write();
            for mat in materials.iter() {
                let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
                material_ids.push(id);
                let resolve_tex =
                    |idx: Option<usize>| idx.and_then(|i| texture_ids.get(i).copied());
                let gpu_data = MaterialGpuData {
                    id,
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
                material_sources.insert(
                    id,
                    MaterialSource {
                        scene_id,
                        data: gpu_data,
                    },
                );
            }
        }

        let mut mesh_ids = Vec::with_capacity(mesh_primitives.len());
        {
            let mut mesh_sources = self.mesh_sources.write();
            for desc in mesh_primitives.iter() {
                let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
                mesh_ids.push(id);
                mesh_sources.insert(
                    id,
                    MeshSource {
                        scene_id,
                        desc: *desc,
                    },
                );
            }
        }

        self.prime_mesh_bounds(&doc_clone, &mesh_primitives, &mesh_ids);

        let scene_nodes = nodes
            .iter()
            .filter_map(|node| {
                let mesh_id = mesh_ids.get(node.primitive_desc_index)?;
                let material_id = material_ids.get(node.material_index)?;
                Some(SceneNode {
                    mesh: Handle::new(*mesh_id),
                    material: Handle::new(*material_id),
                    transform: node.transform,
                })
            })
            .collect();

        self.scenes
            .write()
            .insert(scene_id, Arc::new(Scene { nodes: scene_nodes }));
    }

    fn release_scene_buffers_if_needed(&self, buffers: &[StreamedBuffer]) {
        if self.scene_buffer_budget != 0 {
            return;
        }
        for buffer in buffers {
            buffer.release_pages();
        }
    }

    fn drop_scene_buffers(&self, scene_id: usize) {
        let mut contexts = self.scene_contexts.write();
        let Some(ctx) = contexts.get_mut(&scene_id) else {
            return;
        };
        if let Some(buffers) = ctx.buffers.take() {
            self.release_scene_buffers_if_needed(buffers.as_ref());
            let bytes = ctx.buffers_bytes;
            ctx.buffers_bytes = 0;
            if bytes > 0 {
                let total = self.scene_buffer_bytes.load(AtomicOrdering::Relaxed);
                self.scene_buffer_bytes
                    .store(total.saturating_sub(bytes), AtomicOrdering::Relaxed);
            }
        }
    }

    fn scene_context_status(&self, scene_id: usize) -> SceneContextStatus {
        let (doc, base_path) = {
            let contexts = self.scene_contexts.read();
            let Some(ctx) = contexts.get(&scene_id) else {
                return SceneContextStatus::Missing;
            };
            (ctx.doc.clone(), ctx.base_path.clone())
        };

        let Some(buffers) = self.try_scene_buffers(scene_id) else {
            return SceneContextStatus::Pending;
        };

        let allow_release = self.scene_buffer_budget == 0;
        SceneContextStatus::Ready {
            doc,
            buffers: SceneBufferGuard::new(buffers, allow_release),
            base_path,
        }
    }

    fn try_scene_buffers(&self, scene_id: usize) -> Option<Arc<Vec<StreamedBuffer>>> {
        {
            let mut contexts = self.scene_contexts.write();
            let ctx = contexts.get_mut(&scene_id)?;
            if let Some(buffers) = ctx.buffers.clone() {
                ctx.buffers_last_used = Instant::now();
                return Some(buffers);
            }
        }

        let _ = self.queue_scene_buffer_load(scene_id);
        None
    }

    fn queue_scene_buffer_load(&self, scene_id: usize) -> bool {
        {
            let inflight = self.scene_buffer_inflight.read();
            if inflight.contains(&scene_id) {
                return true;
            }
        }

        let (doc, base_path, scene_path) = {
            let contexts = self.scene_contexts.read();
            let Some(ctx) = contexts.get(&scene_id) else {
                return false;
            };
            (
                ctx.doc.clone(),
                ctx.base_path.clone(),
                ctx.scene_path.clone(),
            )
        };

        let request = AssetLoadRequest::SceneBuffers {
            scene_id,
            doc,
            base_path,
            scene_path,
        };
        if !self.enqueue_worker_request(request) {
            return false;
        }

        self.scene_buffer_inflight.write().insert(scene_id);
        true
    }

    fn scene_has_streaming_activity(&self, scene_id: usize) -> bool {
        if self.scene_buffer_inflight.read().contains(&scene_id) {
            return true;
        }

        let inflight: Vec<(AssetStreamKind, usize)> = {
            let inflight = self.streaming_inflight.read();
            inflight.values().map(|req| (req.kind, req.id)).collect()
        };
        let backlog: Vec<(AssetStreamKind, usize)> = {
            let backlog = self.streaming_backlog.read();
            backlog.iter().map(|req| (req.kind, req.id)).collect()
        };

        if inflight.is_empty() && backlog.is_empty() {
            return false;
        }

        let mesh_sources = self.mesh_sources.read();
        let texture_sources = self.texture_sources.read();
        for (kind, id) in inflight.into_iter().chain(backlog) {
            match kind {
                AssetStreamKind::Mesh => {
                    if let Some(source) = mesh_sources.get(&id) {
                        if source.scene_id == scene_id {
                            return true;
                        }
                    }
                }
                AssetStreamKind::Texture => {
                    if let Some(source) = texture_sources.get(&id) {
                        if source.scene_id == scene_id {
                            return true;
                        }
                    }
                }
                AssetStreamKind::Material => {}
            }
        }

        false
    }

    fn active_streaming_scene_ids(&self) -> HashSet<usize> {
        let mut active: HashSet<usize> = HashSet::new();
        {
            let inflight = self.scene_buffer_inflight.read();
            for scene_id in inflight.iter() {
                active.insert(*scene_id);
            }
        }

        let inflight: Vec<(AssetStreamKind, usize)> = {
            let inflight = self.streaming_inflight.read();
            inflight.values().map(|req| (req.kind, req.id)).collect()
        };
        let backlog: Vec<(AssetStreamKind, usize)> = {
            let backlog = self.streaming_backlog.read();
            backlog.iter().map(|req| (req.kind, req.id)).collect()
        };

        if inflight.is_empty() && backlog.is_empty() {
            return active;
        }

        let mesh_sources = self.mesh_sources.read();
        let texture_sources = self.texture_sources.read();
        for (kind, id) in inflight.into_iter().chain(backlog) {
            match kind {
                AssetStreamKind::Mesh => {
                    if let Some(source) = mesh_sources.get(&id) {
                        active.insert(source.scene_id);
                    }
                }
                AssetStreamKind::Texture => {
                    if let Some(source) = texture_sources.get(&id) {
                        active.insert(source.scene_id);
                    }
                }
                AssetStreamKind::Material => {}
            }
        }

        active
    }

    fn mark_scene_asset_complete(&self, scene_id: usize) {
        let budget = self.scene_buffer_budget;
        let keep_buffers = budget == 0 && self.scene_has_streaming_activity(scene_id);
        let mut contexts = self.scene_contexts.write();
        let Some(ctx) = contexts.get_mut(&scene_id) else {
            return;
        };
        if ctx.pending_assets > 0 {
            ctx.pending_assets -= 1;
        }
        if budget == 0 && ctx.pending_assets == 0 && ctx.buffers.is_some() {
            if keep_buffers {
                return;
            }
            if let Some(buffers) = ctx.buffers.take() {
                self.release_scene_buffers_if_needed(buffers.as_ref());
                let bytes = ctx.buffers_bytes;
                ctx.buffers_bytes = 0;
                if bytes > 0 {
                    let total = self.scene_buffer_bytes.load(AtomicOrdering::Relaxed);
                    self.scene_buffer_bytes
                        .store(total.saturating_sub(bytes), AtomicOrdering::Relaxed);
                }
            }
        }
    }

    fn prime_mesh_bounds(
        &self,
        doc: &gltf::Document,
        mesh_primitives: &[MeshPrimitiveDesc],
        mesh_ids: &[usize],
    ) {
        let mut map = self.mesh_aabb_map.write();
        for (idx, desc) in mesh_primitives.iter().enumerate() {
            if let Some(mesh_id) = mesh_ids.get(idx) {
                let bounds = estimate_primitive_bounds(doc, desc).unwrap_or(Aabb {
                    min: Vec3::ZERO,
                    max: Vec3::ZERO,
                });
                map.0.insert(*mesh_id, bounds);
            }
        }
    }

    fn consume_budget(budget: &mut usize) {
        if *budget > 0 {
            *budget -= 1;
        }
    }

    fn merge_stream_request(
        existing: &mut AssetStreamingRequest,
        incoming: &AssetStreamingRequest,
    ) {
        existing.priority = existing.priority.max(incoming.priority);
        existing.max_lod = match (existing.max_lod, incoming.max_lod) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (None, _) | (_, None) => None,
        };
        existing.force_low_res &= incoming.force_low_res;
    }

    fn update_streaming_inflight(&self, request: &AssetStreamingRequest) {
        let mut inflight = self.streaming_inflight.write();
        inflight
            .entry((request.kind, request.id))
            .and_modify(|existing| Self::merge_stream_request(existing, request))
            .or_insert_with(|| request.clone());
    }

    fn take_streaming_inflight(
        &self,
        kind: AssetStreamKind,
        id: usize,
    ) -> Option<AssetStreamingRequest> {
        self.streaming_inflight.write().remove(&(kind, id))
    }

    fn is_streaming_inflight(&self, kind: AssetStreamKind, id: usize) -> bool {
        self.streaming_inflight.read().contains_key(&(kind, id))
    }

    fn requeue_stream_request(&self, request: AssetStreamingRequest) {
        let mut backlog = self.streaming_backlog.write();
        backlog.push_front(request);
        // Requeued requests already went out to the renderer; keep them to avoid inflight stalls.
    }

    fn queue_low_res_generation(
        &self,
        id: usize,
        name: String,
        kind: AssetKind,
        data: Arc<[u8]>,
        format: wgpu::TextureFormat,
        dimensions: (u32, u32),
    ) {
        let mut inflight = self.low_res_inflight.write();
        if !inflight.insert(id) {
            return;
        }
        let request = AssetLoadRequest::LowResTexture {
            id,
            name,
            kind,
            data,
            format,
            dimensions,
            max_dim: self.asset_streaming_tuning.low_res_max_dim,
        };
        if !self.enqueue_worker_request(request) {
            inflight.remove(&id);
            warn!(
                "Failed to queue low-res texture request {}; worker thread offline",
                id
            );
        }
    }

    fn enqueue_stream_requests(&self) {
        let mut backlog = self.streaming_backlog.write();
        let mut inflight = self.streaming_inflight.write();
        let mut merged: HashMap<(AssetStreamKind, usize), AssetStreamingRequest> = HashMap::new();
        let mut had_new = false;

        let plan_len = self.latest_streaming_plan.read().len();
        let limit = self.streaming_backlog_limit.max(plan_len);
        while let Ok(mut req) = self.stream_request_receiver.try_recv() {
            had_new = true;
            let boost = self.plan_priority(req.kind, req.id);
            req.priority = req.priority.max(boost);
            if let Some(existing) = inflight.get_mut(&(req.kind, req.id)) {
                Self::merge_stream_request(existing, &req);
                continue;
            }
            merged
                .entry((req.kind, req.id))
                .and_modify(|existing| Self::merge_stream_request(existing, &req))
                .or_insert(req);
        }

        if !had_new {
            return;
        }

        if limit == 0 {
            return;
        }

        for mut req in backlog.drain(..) {
            let boost = self.plan_priority(req.kind, req.id);
            req.priority = req.priority.max(boost);
            if let Some(existing) = inflight.get_mut(&(req.kind, req.id)) {
                Self::merge_stream_request(existing, &req);
                continue;
            }
            merged
                .entry((req.kind, req.id))
                .and_modify(|existing| Self::merge_stream_request(existing, &req))
                .or_insert(req);
        }

        let mut merged_vec: Vec<_> = merged.into_values().collect();
        merged_vec.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
                .then_with(|| Self::stream_kind_rank(a.kind).cmp(&Self::stream_kind_rank(b.kind)))
        });
        if merged_vec.len() > limit {
            merged_vec.truncate(limit);
        }
        *backlog = VecDeque::from(merged_vec);
    }

    fn process_stream_requests(&self, limit: usize, creation_budget: &mut usize) {
        self.enqueue_stream_requests();

        let mut backlog = self.streaming_backlog.write();
        let mut sent = 0usize;
        let mut attempts = 0usize;
        let total = backlog.len();

        while sent < limit && attempts < total {
            let Some(request) = backlog.pop_front() else {
                break;
            };
            attempts += 1;
            if self.dispatch_stream_request(&request, creation_budget) {
                sent += 1;
            } else {
                backlog.push_back(request);
            }
        }
    }

    fn process_reupload_queue(&self, limit: usize) -> usize {
        if limit == 0 {
            return 0;
        }
        let mut queue = self.reupload_queue.write();
        let mut sent = 0usize;
        let mut creation_budget = 0usize;

        while sent < limit {
            let Some(request) = queue.pop_front() else {
                break;
            };
            if self.dispatch_stream_request(&request, &mut creation_budget) {
                sent += 1;
            } else {
                queue.push_front(request);
                break;
            }
        }

        sent
    }

    fn dispatch_stream_request(
        &self,
        request: &AssetStreamingRequest,
        creation_budget: &mut usize,
    ) -> bool {
        match request.kind {
            AssetStreamKind::Mesh => {
                let mesh_opt = self.mesh_cache.read().get(&request.id).cloned();
                if let Some(mesh) = mesh_opt {
                    let lods = {
                        let lods = mesh.lods.read();
                        select_mesh_lod_payloads(&lods, request.max_lod)
                    };

                    if lods.is_empty() {
                        return false;
                    }
                    self.sync_mesh_cache_size(request.id, &mesh);

                    match self.try_send_asset_message(RenderMessage::CreateMesh {
                        id: request.id,
                        total_lods: mesh.total_lods,
                        lods,
                        bounds: mesh.bounds,
                    }) {
                        Ok(()) => {
                            self.mesh_aabb_map.write().0.insert(request.id, mesh.bounds);
                            self.touch_cache_entry(AssetStreamKind::Mesh, request.id);
                            return true;
                        }
                        Err(AssetSendError::Budget) | Err(AssetSendError::Full) => return false,
                        Err(AssetSendError::Disconnected) => {
                            warn!(
                                "Failed to send stream mesh {}; render thread offline",
                                request.id
                            );
                            return false;
                        }
                    }
                }

                if *creation_budget == 0 {
                    return false;
                }

                if self.is_streaming_inflight(AssetStreamKind::Mesh, request.id) {
                    self.update_streaming_inflight(request);
                    return true;
                }

                let pending = match self.mesh_sources.read().get(&request.id).copied() {
                    Some(p) => p,
                    None => {
                        warn!(
                            "Streaming request for mesh {} ignored; mesh not available",
                            request.id
                        );
                        return false;
                    }
                };

                let (doc, buffers) = match self.scene_context_status(pending.scene_id) {
                    SceneContextStatus::Ready { doc, buffers, .. } => (doc, buffers),
                    SceneContextStatus::Pending => return false,
                    SceneContextStatus::Missing => {
                        warn!(
                            "Streaming request for mesh {} missing scene context",
                            request.id
                        );
                        self.mark_scene_asset_complete(pending.scene_id);
                        Self::consume_budget(creation_budget);
                        return true;
                    }
                };
                self.update_streaming_inflight(request);
                let queued = self.enqueue_worker_request(AssetLoadRequest::StreamMesh {
                    id: request.id,
                    scene_id: pending.scene_id,
                    desc: pending.desc,
                    doc,
                    buffers,
                    tuning: self.asset_streaming_tuning,
                });
                if !queued {
                    self.take_streaming_inflight(AssetStreamKind::Mesh, request.id);
                    self.mark_scene_asset_complete(pending.scene_id);
                    return true;
                }
                Self::consume_budget(creation_budget);
                true
            }
            AssetStreamKind::Material => {
                let material_opt = self.material_cache.read().get(&request.id).cloned();
                if let Some(material) = material_opt {
                    match self
                        .try_send_asset_message(RenderMessage::CreateMaterial((*material).clone()))
                    {
                        Ok(()) => {
                            self.touch_cache_entry(AssetStreamKind::Material, request.id);
                            return true;
                        }
                        Err(AssetSendError::Budget) | Err(AssetSendError::Full) => return false,
                        Err(AssetSendError::Disconnected) => {
                            warn!(
                                "Failed to send stream material {}; render thread offline",
                                request.id
                            );
                            return false;
                        }
                    }
                }

                let pending = match self.material_sources.read().get(&request.id).cloned() {
                    Some(p) => p,
                    None => {
                        warn!(
                            "Streaming request for material {} ignored; material not available",
                            request.id
                        );
                        return false;
                    }
                };

                if self.material_budget > 0 {
                    let size_bytes = std::mem::size_of_val(&pending.data);
                    self.material_cache
                        .write()
                        .insert(request.id, Arc::new(pending.data.clone()));
                    self.record_cache_entry(AssetStreamKind::Material, request.id, size_bytes);
                    self.enforce_cache_budget(AssetStreamKind::Material);
                }
                self.mark_scene_asset_complete(pending.scene_id);
                match self
                    .try_send_asset_message(RenderMessage::CreateMaterial(pending.data.clone()))
                {
                    Ok(()) => true,
                    Err(AssetSendError::Budget) | Err(AssetSendError::Full) => false,
                    Err(AssetSendError::Disconnected) => {
                        warn!(
                            "Failed to send streamed material {}; render thread offline",
                            request.id
                        );
                        false
                    }
                }
            }
            AssetStreamKind::Texture => {
                let texture_opt = self.texture_cache.read().get(&request.id).cloned();
                if let Some(tex) = texture_opt {
                    if request.force_low_res {
                        if let Some(low) = tex.low_res.read().clone() {
                            match self.try_send_asset_message(RenderMessage::CreateTexture {
                                id: request.id,
                                name: tex.name.clone(),
                                kind: tex.kind,
                                data: low.data.clone(),
                                format: low.format,
                                width: low.dimensions.0,
                                height: low.dimensions.1,
                            }) {
                                Ok(()) => {}
                                Err(AssetSendError::Budget) | Err(AssetSendError::Full) => {
                                    return false;
                                }
                                Err(AssetSendError::Disconnected) => {
                                    warn!(
                                        "Failed to send low-res texture {}; render thread offline",
                                        request.id
                                    );
                                    return false;
                                }
                            }
                        } else {
                            self.queue_low_res_generation(
                                request.id,
                                tex.name.clone(),
                                tex.kind,
                                tex.data.clone(),
                                tex.format,
                                tex.dimensions,
                            );
                            self.touch_cache_entry(AssetStreamKind::Texture, request.id);
                            return true;
                        }
                    } else {
                        match self.try_send_asset_message(RenderMessage::CreateTexture {
                            id: request.id,
                            name: tex.name.clone(),
                            kind: tex.kind,
                            data: tex.data.clone(),
                            format: tex.format,
                            width: tex.dimensions.0,
                            height: tex.dimensions.1,
                        }) {
                            Ok(()) => {}
                            Err(AssetSendError::Budget) | Err(AssetSendError::Full) => {
                                return false;
                            }
                            Err(AssetSendError::Disconnected) => {
                                warn!(
                                    "Failed to send stream texture {}; render thread offline",
                                    request.id
                                );
                                return false;
                            }
                        }
                    }
                    self.touch_cache_entry(AssetStreamKind::Texture, request.id);
                    return true;
                }

                if *creation_budget == 0 {
                    return false;
                }

                if self.is_streaming_inflight(AssetStreamKind::Texture, request.id) {
                    self.update_streaming_inflight(request);
                    return true;
                }

                let pending = match self.texture_sources.read().get(&request.id).copied() {
                    Some(p) => p,
                    None => {
                        warn!(
                            "Streaming request for texture {} ignored; texture not available",
                            request.id
                        );
                        return false;
                    }
                };

                let (doc, buffers, base_path) = match self.scene_context_status(pending.scene_id) {
                    SceneContextStatus::Ready {
                        doc,
                        buffers,
                        base_path,
                    } => (doc, buffers, base_path),
                    SceneContextStatus::Pending => return false,
                    SceneContextStatus::Missing => {
                        warn!(
                            "Streaming request for texture {} missing scene context",
                            request.id
                        );
                        self.mark_scene_asset_complete(pending.scene_id);
                        Self::consume_budget(creation_budget);
                        return true;
                    }
                };
                self.update_streaming_inflight(request);
                let queued = self.enqueue_worker_request(AssetLoadRequest::StreamTexture {
                    id: request.id,
                    scene_id: pending.scene_id,
                    tex_index: pending.request.tex_index,
                    kind: pending.request.kind,
                    doc,
                    buffers,
                    base_path,
                });
                if !queued {
                    self.take_streaming_inflight(AssetStreamKind::Texture, request.id);
                    self.mark_scene_asset_complete(pending.scene_id);
                    return true;
                }
                Self::consume_budget(creation_budget);
                true
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
        AssetLoadRequest::SceneBuffers { scene_path, .. } => {
            scene_path.as_deref().unwrap_or_else(|| Path::new(""))
        }
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

fn is_gltf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("gltf"))
        .unwrap_or(false)
}

fn load_gltf_streaming(path: &Path) -> Result<(gltf::Document, Vec<StreamedBuffer>), String> {
    let gltf = gltf::Gltf::open(path).map_err(|e| e.to_string())?;
    let mut blob = gltf.blob;
    let document = gltf.document;
    let base_dir = path.parent();
    let mut buffers = Vec::new();

    for buffer in document.buffers() {
        let data = match buffer.source() {
            gltf::buffer::Source::Bin => {
                let blob_data = blob
                    .take()
                    .ok_or_else(|| format!("Missing BIN chunk for '{}'", path.display()))?;
                StreamedBuffer::owned(blob_data)
            }
            gltf::buffer::Source::Uri(uri) => {
                if let Some((data, _)) = decode_data_uri(uri)? {
                    StreamedBuffer::owned(data)
                } else {
                    let base = base_dir.ok_or_else(|| {
                        format!(
                            "Cannot resolve buffer URI '{}' without a parent directory (file: '{}')",
                            uri,
                            path.display()
                        )
                    })?;
                    let buf_path = base.join(uri);
                    let file = File::open(&buf_path).map_err(|e| {
                        format!("Failed to open buffer '{}': {}", buf_path.display(), e)
                    })?;
                    let map = unsafe {
                        MmapOptions::new().map(&file).map_err(|e| {
                            format!("Failed to mmap buffer '{}': {}", buf_path.display(), e)
                        })?
                    };
                    StreamedBuffer::mapped(map)
                }
            }
        };
        if data.len() < buffer.length() {
            return Err(format!(
                "Buffer {} shorter than declared length ({} < {})",
                buffer.index(),
                data.len(),
                buffer.length()
            ));
        }
        buffers.push(data);
    }

    Ok((document, buffers))
}

fn load_scene_buffers(
    doc: &gltf::Document,
    scene_path: Option<&Path>,
    base_path: Option<&Path>,
) -> Result<Vec<StreamedBuffer>, String> {
    let mut buffers = Vec::new();
    let mut blob: Option<Vec<u8>> = None;
    let base_dir = base_path.or_else(|| scene_path.and_then(|p| p.parent()));

    for buffer in doc.buffers() {
        let data = match buffer.source() {
            gltf::buffer::Source::Bin => {
                if blob.is_none() {
                    let path = scene_path.ok_or_else(|| {
                        "Cannot resolve BIN buffer without a scene file path".to_string()
                    })?;
                    let gltf = gltf::Gltf::open(path).map_err(|e| e.to_string())?;
                    blob = gltf.blob;
                }
                let blob_data = blob
                    .take()
                    .ok_or_else(|| format!("Missing BIN chunk for '{}'", buffer.index()))?;
                StreamedBuffer::owned(blob_data)
            }
            gltf::buffer::Source::Uri(uri) => {
                if let Some((data, _)) = decode_data_uri(uri)? {
                    StreamedBuffer::owned(data)
                } else {
                    let base = base_dir.ok_or_else(|| {
                        format!(
                            "Cannot resolve buffer URI '{}' without a base directory",
                            uri
                        )
                    })?;
                    let buf_path = base.join(uri);
                    let file = File::open(&buf_path).map_err(|e| {
                        format!("Failed to open buffer '{}': {}", buf_path.display(), e)
                    })?;
                    let map = unsafe {
                        MmapOptions::new().map(&file).map_err(|e| {
                            format!("Failed to mmap buffer '{}': {}", buf_path.display(), e)
                        })?
                    };
                    StreamedBuffer::mapped(map)
                }
            }
        };
        if data.len() < buffer.length() {
            return Err(format!(
                "Buffer {} shorter than declared length ({} < {})",
                buffer.index(),
                data.len(),
                buffer.length()
            ));
        }
        buffers.push(data);
    }

    Ok(buffers)
}

fn parse_mesh_from_gltf_path(path: &Path) -> Result<(Vec<Vertex>, Vec<u32>, Aabb), String> {
    let (document, buffers) = load_gltf_streaming(path)?;
    parse_mesh_document(&document, &buffers)
}

fn parse_scene_from_gltf_path(path: &Path) -> Result<ParsedGltfScene, String> {
    let gltf = gltf::Gltf::open(path).map_err(|e| e.to_string())?;
    parse_scene_document(gltf.document, None, path.parent(), Some(path.to_path_buf()))
}

fn encode_basis_with_mips(
    rgba: &[u8],
    width: u32,
    height: u32,
    kind: AssetKind,
) -> Result<Vec<u8>, String> {
    let expected_len = width.saturating_mul(height).saturating_mul(4) as usize;
    if rgba.len() != expected_len {
        return Err("Basis mip generation input size mismatch.".to_string());
    }

    let mut params = CompressorParams::new();
    params.set_generate_mipmaps(true);
    params.set_mipmap_smallest_dimension(1);
    params.set_basis_format(BasisTextureFormat::UASTC4x4);
    params.set_uastc_quality_level(UASTC_QUALITY_MIN);

    match kind {
        AssetKind::Normal => params.tune_for_normal_maps(),
        AssetKind::Albedo | AssetKind::Emission => params.set_color_space(ColorSpace::Srgb),
        _ => params.set_color_space(ColorSpace::Linear),
    }

    let mut image = params.source_image_mut(0);
    image.init(rgba, width, height, 4);

    let thread_count = std::thread::available_parallelism()
        .map(|count| count.get() as u32)
        .unwrap_or(1)
        .min(4)
        .max(1);
    let mut compressor = Compressor::new(thread_count);
    if !unsafe { compressor.init(&params) } {
        return Err("Failed to initialize Basis compressor.".to_string());
    }
    unsafe {
        compressor
            .process()
            .map_err(|e| format!("Basis encode failed: {:?}", e))?;
    }

    Ok(compressor.basis_file().to_vec())
}

fn decode_ktx2(
    bytes: &[u8],
    kind: AssetKind,
) -> Result<(Vec<u8>, wgpu::TextureFormat, (u32, u32)), String> {
    let reader = Reader::new(bytes).map_err(|e| e.to_string())?;
    let header = reader.header();
    let mut dimensions = (header.pixel_width, header.pixel_height);
    let forward_ta = std::env::var("HELMER_PATH") == Ok("forwardTA".to_string());

    // Handle uncompressed formats first
    if let Some(format) = header.format {
        let wgpu_format = match format {
            ktx2::Format::R8G8B8A8_UNORM => wgpu::TextureFormat::Rgba8Unorm,
            ktx2::Format::R8G8B8A8_SRGB => wgpu::TextureFormat::Rgba8UnormSrgb,
            _ => return Err(format!("Unsupported direct KTX2 format: {:?}", format)),
        };
        let level_count = header.level_count.max(1);
        let mut data = if level_count > 1 && !forward_ta {
            let mut combined = Vec::new();
            for level in reader.levels() {
                combined.extend_from_slice(level.data);
            }
            combined
        } else {
            let level_data = reader.levels().next().ok_or("No image levels found")?;
            level_data.data.to_vec()
        };

        if forward_ta {
            if dimensions.0 != FORWARD_TA_TARGET_RES || dimensions.1 != FORWARD_TA_TARGET_RES {
                // We assume R8G8B8A8 format here, which is 4 bytes per pixel.
                if let Some(image_buffer) = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                    dimensions.0,
                    dimensions.1,
                    data.clone(),
                ) {
                    let resized = image::imageops::resize(
                        &image_buffer,
                        FORWARD_TA_TARGET_RES,
                        FORWARD_TA_TARGET_RES,
                        image::imageops::FilterType::Lanczos3,
                    );
                    data = resized.into_raw();
                    dimensions = (FORWARD_TA_TARGET_RES, FORWARD_TA_TARGET_RES);
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

    let target_transcode_format = TranscoderTextureFormat::BC7_RGBA;
    let target_wgpu_format = match kind {
        AssetKind::Albedo | AssetKind::Emission => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
        _ => wgpu::TextureFormat::Bc7RgbaUnorm,
    };

    let transcode_params = TranscodeParameters {
        level_index: 0,
        ..Default::default()
    };

    if forward_ta {
        let base_rgba = transcoder
            .transcode_image_level(bytes, TranscoderTextureFormat::RGBA32, transcode_params)
            .map_err(|e| format!("Failed to transcode KTX2 to RGBA: {:?}", e))?;
        transcoder.end_transcoding();
        let mut data = base_rgba;

        if dimensions.0 != FORWARD_TA_TARGET_RES || dimensions.1 != FORWARD_TA_TARGET_RES {
            let image_buffer = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                dimensions.0,
                dimensions.1,
                data,
            )
            .ok_or_else(|| "Failed to create image buffer from transcoded KTX2 data".to_string())?;

            let resized = image::imageops::resize(
                &image_buffer,
                FORWARD_TA_TARGET_RES,
                FORWARD_TA_TARGET_RES,
                image::imageops::FilterType::Lanczos3,
            );
            data = resized.into_raw();
            dimensions = (FORWARD_TA_TARGET_RES, FORWARD_TA_TARGET_RES);
        }

        let final_format = match kind {
            AssetKind::Albedo | AssetKind::Emission => wgpu::TextureFormat::Rgba8UnormSrgb,
            _ => wgpu::TextureFormat::Rgba8Unorm,
        };
        return Ok((data, final_format, dimensions));
    }

    let level_count = transcoder.image_level_count(bytes, 0).max(1);
    if level_count > 1 {
        let mut combined = Vec::new();
        for level in 0..level_count {
            let level_data = transcoder
                .transcode_image_level(
                    bytes,
                    target_transcode_format,
                    TranscodeParameters {
                        level_index: level,
                        ..Default::default()
                    },
                )
                .map_err(|e| format!("Failed to transcode KTX2 image level {}: {:?}", level, e))?;
            combined.extend_from_slice(&level_data);
        }
        transcoder.end_transcoding();
        return Ok((combined, target_wgpu_format, dimensions));
    }

    let base_rgba = transcoder
        .transcode_image_level(bytes, TranscoderTextureFormat::RGBA32, transcode_params)
        .map_err(|e| {
            format!(
                "Failed to transcode KTX2 to RGBA for mip generation: {:?}",
                e
            )
        })?;
    transcoder.end_transcoding();

    let basis_data = encode_basis_with_mips(&base_rgba, dimensions.0, dimensions.1, kind)?;

    let mut mip_transcoder = Transcoder::new();
    if mip_transcoder.prepare_transcoding(&basis_data).is_err() {
        return Err("Failed to prepare Basis transcoder for mip generation.".to_string());
    }

    let mip_level_count = mip_transcoder.image_level_count(&basis_data, 0).max(1);
    let mut combined = Vec::new();
    for level in 0..mip_level_count {
        let level_data = mip_transcoder
            .transcode_image_level(
                &basis_data,
                target_transcode_format,
                TranscodeParameters {
                    level_index: level,
                    ..Default::default()
                },
            )
            .map_err(|e| format!("Failed to transcode generated mip level {}: {:?}", level, e))?;
        combined.extend_from_slice(&level_data);
    }
    mip_transcoder.end_transcoding();

    Ok((combined, target_wgpu_format, dimensions))
}

fn parse_ron_material(bytes: &[u8]) -> Result<MaterialFile, String> {
    ron::de::from_bytes(bytes).map_err(|e| e.to_string())
}

fn meshopt_enabled() -> bool {
    std::env::var("HELMER_ENABLE_MESHOPT")
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(true)
}

fn mesh_size_bytes(mesh: &Mesh) -> usize {
    let lods = mesh.lods.read();
    let mut total = 0usize;
    for lod in lods.iter() {
        total = total.saturating_add(lod.vertices.len() * std::mem::size_of::<Vertex>());
        total = total.saturating_add(lod.indices.len() * std::mem::size_of::<u32>());
        total = total.saturating_add(meshlet_lod_size_bytes(&lod.meshlets));
    }
    total
}

fn optimize_base_lod(vertices: &[Vertex], indices: &[u32]) -> Vec<u32> {
    if meshopt_enabled() {
        optimize_vertex_cache(indices, vertices.len())
    } else {
        indices.to_vec()
    }
}

fn generate_lods(
    vertices: &[Vertex],
    indices: &[u32],
    tuning: &AssetStreamingTuning,
) -> Vec<Vec<u32>> {
    let mut lods = Vec::new();
    if indices.is_empty() || vertices.is_empty() {
        return lods;
    }

    // Allow opting out of meshopt paths entirely for robustness.
    let use_meshopt = meshopt_enabled();
    if !use_meshopt {
        lods.push(indices.to_vec());
        return lods;
    }

    // Avoid calling into meshopt on extremely large meshes; keep original indices only.
    let too_large = vertices.len() > tuning.lod_safe_vertex_limit
        || indices.len() > tuning.lod_safe_index_limit;

    // LOD 0: The original mesh, optimized for the GPU vertex cache.
    let lod0 = if too_large {
        indices.to_vec()
    } else {
        optimize_vertex_cache(indices, vertices.len())
    };
    lods.push(lod0);

    if too_large {
        return lods;
    }

    // Prepare the vertex data adapter for the simplifier.
    // The adapter tells meshopt how to access the position data from your Vertex struct.
    let vertex_data_adapter = match VertexDataAdapter::new(
        bytemuck::cast_slice(vertices), // The raw vertex buffer as a byte slice
        std::mem::size_of::<Vertex>(),  // The stride (size of one vertex)
        0, // The offset of the position attribute (it's the first field)
    ) {
        Ok(adapter) => adapter,
        Err(e) => {
            warn!(
                "Failed to create vertex adapter for LOD generation: {:?}",
                e
            );
            return lods;
        }
    };

    // Generate lower detail LODs from the original index buffer.
    let simplification_error = tuning.lod_simplification_error.max(0.0);

    for &ratio in &tuning.lod_targets {
        if ratio <= 0.0 || ratio >= 1.0 {
            continue;
        }
        let target_index_count = (indices.len() as f32 * ratio) as usize;
        let target_index_count = target_index_count - (target_index_count % 3);

        if target_index_count < indices.len() {
            // Always simplify from the original, high-quality index buffer.
            let simplified_indices =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    simplify(
                        indices,
                        &vertex_data_adapter,
                        target_index_count,
                        simplification_error,
                        SimplifyOptions::all(),
                        None,
                    )
                })) {
                    Ok(data) => data,
                    Err(_) => {
                        warn!("meshopt simplification panicked; keeping previous LODs only.");
                        break;
                    }
                };

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

struct MeshPayload {
    lods: Vec<MeshLodPayload>,
    total_lods: usize,
    bounds: Aabb,
}

fn compact_mesh_lod(
    vertices: &[Vertex],
    indices: &[u32],
    lod_index: usize,
) -> Option<MeshLodPayload> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    let mut remap = vec![u32::MAX; vertices.len()];
    let mut compact_vertices: Vec<Vertex> = Vec::new();
    let mut compact_indices: Vec<u32> = Vec::with_capacity(indices.len());

    for &idx in indices {
        let src = idx as usize;
        if src >= vertices.len() {
            return None;
        }
        let mapped = remap[src];
        let new_index = if mapped == u32::MAX {
            let new_index = compact_vertices.len() as u32;
            remap[src] = new_index;
            compact_vertices.push(vertices[src]);
            new_index
        } else {
            mapped
        };
        compact_indices.push(new_index);
    }

    if compact_vertices.is_empty() || compact_indices.is_empty() {
        return None;
    }

    let meshlets = build_meshlet_lod(&compact_vertices, &compact_indices);
    Some(MeshLodPayload {
        lod_index,
        vertices: Arc::from(compact_vertices),
        indices: Arc::from(compact_indices),
        meshlets,
    })
}

fn build_mesh_payload(
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    bounds: Aabb,
    tuning: &AssetStreamingTuning,
) -> Option<MeshPayload> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }
    let lod_indices = generate_lods(&vertices, &indices, tuning);
    if lod_indices.is_empty() {
        return None;
    }
    let mut lods = Vec::with_capacity(lod_indices.len());
    for (lod_index, lod_indices) in lod_indices.iter().enumerate() {
        if let Some(lod) = compact_mesh_lod(&vertices, lod_indices, lod_index) {
            lods.push(lod);
        }
    }
    if lods.is_empty() {
        return None;
    }
    Some(MeshPayload {
        total_lods: lods.len(),
        lods,
        bounds,
    })
}

fn estimate_primitive_bounds(doc: &gltf::Document, desc: &MeshPrimitiveDesc) -> Option<Aabb> {
    let mesh = doc.meshes().nth(desc.mesh_index)?;
    let primitive = mesh.primitives().nth(desc.primitive_index)?;
    let accessor = primitive.get(&gltf::Semantic::Positions)?;
    let min_val = accessor.min()?;
    let max_val = accessor.max()?;
    let min_vals = min_val.as_array()?;
    let max_vals = max_val.as_array()?;

    let min = Vec3::new(
        min_vals.get(0)?.as_f64()? as f32,
        min_vals.get(1)?.as_f64()? as f32,
        min_vals.get(2)?.as_f64()? as f32,
    );
    let max = Vec3::new(
        max_vals.get(0)?.as_f64()? as f32,
        max_vals.get(1)?.as_f64()? as f32,
        max_vals.get(2)?.as_f64()? as f32,
    );

    Some(Aabb { min, max })
}

fn select_mesh_lod_payloads(
    lods: &[MeshLodPayload],
    max_lod: Option<usize>,
) -> Vec<MeshLodPayload> {
    if lods.is_empty() {
        return Vec::new();
    }
    let target = max_lod.unwrap_or(0);
    if let Some(lod) = lods.iter().find(|lod| lod.lod_index == target) {
        return vec![lod.clone()];
    }

    let mut best: Option<&MeshLodPayload> = None;
    for lod in lods.iter() {
        if lod.lod_index <= target {
            best = Some(lod);
        }
    }
    if let Some(lod) = best {
        vec![lod.clone()]
    } else {
        vec![lods[0].clone()]
    }
}

fn triangulate_strip(indices: &[u32]) -> Vec<u32> {
    if indices.len() < 3 {
        return Vec::new();
    }
    let mut tris = Vec::with_capacity((indices.len() - 2) * 3);
    for i in 0..(indices.len() - 2) {
        let i0 = indices[i];
        let i1 = indices[i + 1];
        let i2 = indices[i + 2];
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }
        if i % 2 == 0 {
            tris.extend_from_slice(&[i0, i1, i2]);
        } else {
            tris.extend_from_slice(&[i1, i0, i2]);
        }
    }
    tris
}

fn triangulate_fan(indices: &[u32]) -> Vec<u32> {
    if indices.len() < 3 {
        return Vec::new();
    }
    let mut tris = Vec::with_capacity((indices.len() - 2) * 3);
    let base = indices[0];
    for i in 1..(indices.len() - 1) {
        let i1 = indices[i];
        let i2 = indices[i + 1];
        if base == i1 || i1 == i2 || base == i2 {
            continue;
        }
        tris.extend_from_slice(&[base, i1, i2]);
    }
    tris
}

fn process_primitive<B: BufferSource>(
    primitive: &gltf::Primitive,
    buffers: &[B],
) -> Option<(Vec<Vertex>, Vec<u32>, Aabb)> {
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|b| b.as_slice()));
    let mode = primitive.mode();
    let position_accessor = primitive.get(&gltf::Semantic::Positions)?;
    let vertex_count = position_accessor.count();

    if let Some(normals_accessor) = primitive.get(&gltf::Semantic::Normals) {
        let normal_count = normals_accessor.count();
        if normal_count != vertex_count {
            warn!(
                "Primitive {} normal count ({}) does not match position count ({}); padding.",
                primitive.index(),
                normal_count,
                vertex_count
            );
        }
    }

    if let Some(tex_accessor) = primitive.get(&gltf::Semantic::TexCoords(0)) {
        let tex_count = tex_accessor.count();
        if tex_count != vertex_count {
            warn!(
                "Primitive {} texcoord count ({}) does not match position count ({}); padding.",
                primitive.index(),
                tex_count,
                vertex_count
            );
        }
    }

    let mut normal_iter = reader.read_normals();
    let mut tex_iter = reader.read_tex_coords(0).map(|tc| tc.into_f32());

    let mut vertices = Vec::with_capacity(vertex_count);
    let mut min_bounds = Vec3::splat(f32::MAX);
    let mut max_bounds = Vec3::splat(f32::MIN);

    let positions_iter = reader.read_positions()?;
    for position in positions_iter {
        let normal = normal_iter
            .as_mut()
            .and_then(|iter| iter.next())
            .unwrap_or([0.0, 1.0, 0.0]);
        let tex_coord = tex_iter
            .as_mut()
            .and_then(|iter| iter.next())
            .unwrap_or([0.0, 0.0]);
        vertices.push(Vertex::new(position, normal, tex_coord, [0.0; 4]));
        let position_vec = Vec3::from(position);
        min_bounds = min_bounds.min(position_vec);
        max_bounds = max_bounds.max(position_vec);
    }

    if vertices.is_empty() {
        return None;
    }
    let mut indices: Vec<u32> = if let Some(read) = reader.read_indices() {
        read.into_u32().collect()
    } else {
        (0..vertex_count as u32).collect()
    };
    if indices.is_empty() {
        return None;
    }
    indices = match mode {
        gltf::mesh::Mode::Triangles => {
            let trimmed = indices.len() - (indices.len() % 3);
            if trimmed == 0 {
                return None;
            }
            if trimmed != indices.len() {
                warn!(
                    "Primitive {} has {} indices (trimming to {}).",
                    primitive.index(),
                    indices.len(),
                    trimmed
                );
                indices.truncate(trimmed);
            }
            indices
        }
        gltf::mesh::Mode::TriangleStrip => triangulate_strip(&indices),
        gltf::mesh::Mode::TriangleFan => triangulate_fan(&indices),
        gltf::mesh::Mode::Points
        | gltf::mesh::Mode::Lines
        | gltf::mesh::Mode::LineStrip
        | gltf::mesh::Mode::LineLoop => {
            warn!(
                "Primitive {} has unsupported mode {:?}; skipping.",
                primitive.index(),
                mode
            );
            return None;
        }
    };
    if indices.is_empty() {
        return None;
    }
    let max_index = indices.iter().copied().max().unwrap_or(0) as usize;
    if max_index >= vertices.len() {
        warn!(
            "Primitive {} index {} out of bounds for {} vertices; skipping.",
            primitive.index(),
            max_index,
            vertices.len()
        );
        return None;
    }
    // Protect against extremely large primitives that could destabilize C FFI paths.
    const MAX_TANGENT_VERTICES: usize = 5_000_000;
    let skip_tangents =
        vertices.len() > MAX_TANGENT_VERTICES || indices.len() > MAX_TANGENT_VERTICES * 3;

    if !skip_tangents {
        generate_tangents_safe(&mut vertices, &indices);
    }

    let bounds = Aabb {
        min: min_bounds,
        max: max_bounds,
    };
    Some((vertices, indices, bounds))
}

fn generate_tangents_safe(vertices: &mut [Vertex], indices: &[u32]) {
    if vertices.is_empty() || indices.len() < 3 {
        return;
    }

    let mut tan1 = vec![[0.0f32; 3]; vertices.len()];
    let mut tan2 = vec![[0.0f32; 3]; vertices.len()];

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }

        let v0 = Vec3::from(vertices[i0].position);
        let v1 = Vec3::from(vertices[i1].position);
        let v2 = Vec3::from(vertices[i2].position);

        let w0 = Vec3::new(vertices[i0].tex_coord[0], vertices[i0].tex_coord[1], 0.0);
        let w1 = Vec3::new(vertices[i1].tex_coord[0], vertices[i1].tex_coord[1], 0.0);
        let w2 = Vec3::new(vertices[i2].tex_coord[0], vertices[i2].tex_coord[1], 0.0);

        let x1 = v1 - v0;
        let x2 = v2 - v0;
        let s1 = w1.x - w0.x;
        let s2 = w2.x - w0.x;
        let t1 = w1.y - w0.y;
        let t2 = w2.y - w0.y;

        let denom = s1 * t2 - s2 * t1;
        if denom.abs() < 1e-8 {
            continue;
        }
        let r = 1.0 / denom;
        let sdir = Vec3::new(
            (t2 * x1.x - t1 * x2.x) * r,
            (t2 * x1.y - t1 * x2.y) * r,
            (t2 * x1.z - t1 * x2.z) * r,
        );
        let tdir = Vec3::new(
            (s1 * x2.x - s2 * x1.x) * r,
            (s1 * x2.y - s2 * x1.y) * r,
            (s1 * x2.z - s2 * x1.z) * r,
        );

        for (i, s, t) in [(i0, sdir, tdir), (i1, sdir, tdir), (i2, sdir, tdir)] {
            tan1[i][0] += s.x;
            tan1[i][1] += s.y;
            tan1[i][2] += s.z;

            tan2[i][0] += t.x;
            tan2[i][1] += t.y;
            tan2[i][2] += t.z;
        }
    }

    for (i, v) in vertices.iter_mut().enumerate() {
        let n = Vec3::from(v.normal);
        let t = Vec3::from(tan1[i]);
        let mut tangent = (t - n * n.dot(t));
        if tangent.length_squared() > 0.0 {
            tangent = tangent.normalize();
        } else {
            tangent = Vec3::new(1.0, 0.0, 0.0);
        }

        let bitan = Vec3::from(tan2[i]);
        let w = if n.cross(t).dot(bitan) < 0.0 {
            -1.0
        } else {
            1.0
        };
        v.tangent = [tangent.x, tangent.y, tangent.z, w];
    }
}

fn parse_glb(bytes: &[u8]) -> Result<(Vec<Vertex>, Vec<u32>, Aabb), String> {
    let (gltf, buffers, _) = gltf::import_slice(bytes).map_err(|e| e.to_string())?;
    parse_mesh_document(&gltf, &buffers)
}

fn parse_scene_glb_path(path: &Path) -> Result<ParsedGltfScene, String> {
    let gltf = gltf::Gltf::open(path).map_err(|e| e.to_string())?;
    parse_scene_document(gltf.document, None, path.parent(), Some(path.to_path_buf()))
}

fn parse_mesh_document<B: BufferSource>(
    doc: &gltf::Document,
    buffers: &[B],
) -> Result<(Vec<Vertex>, Vec<u32>, Aabb), String> {
    let mut all_vertices: Vec<Vertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();

    for mesh in doc.meshes() {
        for primitive in mesh.primitives() {
            if let Some((mut vertices, indices, _)) = process_primitive(&primitive, buffers) {
                let index_offset = all_vertices.len() as u32;
                all_vertices.append(&mut vertices);
                all_indices.extend(indices.iter().map(|i| i + index_offset));
            }
        }
    }

    if all_vertices.is_empty() {
        return Err("GLTF contains no valid mesh primitives.".to_string());
    }
    let bounds = Aabb::calculate(&all_vertices);
    let base = optimize_base_lod(&all_vertices, &all_indices);
    Ok((all_vertices, base, bounds))
}

fn parse_scene_document(
    doc: gltf::Document,
    buffers: Option<Vec<StreamedBuffer>>,
    base_path: Option<&Path>,
    scene_path: Option<PathBuf>,
) -> Result<ParsedGltfScene, String> {
    let doc_arc = Arc::new(doc);
    let buffers_arc = buffers.map(Arc::new);
    let buffers_bytes = buffers_arc
        .as_ref()
        .map(|buffers| buffers.iter().map(|buf| buf.len()).sum())
        .unwrap_or(0);
    let mut textures: Vec<TextureRequest> = Vec::new();
    let mut texture_lookup: HashMap<(usize, AssetKind), usize> = HashMap::new();
    let mut materials: Vec<IntermediateMaterial> = Vec::new();
    let mut material_lookup: HashMap<Option<usize>, usize> = HashMap::new();
    let mut mesh_primitives: Vec<MeshPrimitiveDesc> = Vec::new();
    let mut nodes: Vec<GltfNodeDesc> = Vec::new();

    materials.push(IntermediateMaterial {
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
    material_lookup.insert(None, 0);

    let mut ensure_texture = |tex: GltfTexture, kind: AssetKind| {
        let key = (tex.index(), kind);
        if let Some(idx) = texture_lookup.get(&key) {
            *idx
        } else {
            let idx = textures.len();
            textures.push(TextureRequest {
                tex_index: tex.index(),
                kind,
            });
            texture_lookup.insert(key, idx);
            idx
        }
    };

    let mut ensure_material = |mat: gltf::Material| {
        let key = mat.index();
        if let Some(existing) = material_lookup.get(&key) {
            return *existing;
        }
        let pbr = mat.pbr_metallic_roughness();
        let mut get_tex =
            |tex_opt: Option<GltfTexture>, kind| tex_opt.map(|t| ensure_texture(t, kind));
        let idx = materials.len();
        materials.push(IntermediateMaterial {
            albedo: pbr.base_color_factor(),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            ao: mat.occlusion_texture().map_or(1.0, |t| t.strength()),
            emission_strength: mat.emissive_strength().unwrap_or(0.0),
            emission_color: mat.emissive_factor(),
            albedo_texture_index: get_tex(
                pbr.base_color_texture().map(|i| i.texture()),
                AssetKind::Albedo,
            ),
            normal_texture_index: get_tex(
                mat.normal_texture().map(|i| i.texture()),
                AssetKind::Normal,
            ),
            metallic_roughness_texture_index: get_tex(
                pbr.metallic_roughness_texture().map(|i| i.texture()),
                AssetKind::MetallicRoughness,
            ),
            emission_texture_index: get_tex(
                mat.emissive_texture().map(|i| i.texture()),
                AssetKind::Emission,
            ),
        });
        material_lookup.insert(key, idx);
        idx
    };

    fn process_node_desc(
        node: &gltf::Node,
        parent_transform: &Mat4,
        mesh_primitives: &mut Vec<MeshPrimitiveDesc>,
        nodes: &mut Vec<GltfNodeDesc>,
        ensure_material: &mut dyn FnMut(gltf::Material) -> usize,
    ) {
        let transform = *parent_transform * Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let primitive_desc_index = mesh_primitives.len();
                let material_index = ensure_material(primitive.material());
                mesh_primitives.push(MeshPrimitiveDesc {
                    mesh_index: mesh.index(),
                    primitive_index: primitive.index(),
                    material_index,
                });
                nodes.push(GltfNodeDesc {
                    primitive_desc_index,
                    material_index,
                    transform,
                });
            }
        }
        for child in node.children() {
            process_node_desc(&child, &transform, mesh_primitives, nodes, ensure_material);
        }
    }

    for scene in doc_arc.scenes() {
        for node in scene.nodes() {
            process_node_desc(
                &node,
                &Mat4::IDENTITY,
                &mut mesh_primitives,
                &mut nodes,
                &mut ensure_material,
            );
        }
    }

    Ok(ParsedGltfScene {
        doc: doc_arc,
        buffers: buffers_arc,
        buffers_bytes,
        base_path: base_path.map(|p| p.to_path_buf()),
        scene_path,
        textures,
        materials,
        mesh_primitives,
        nodes,
    })
}
