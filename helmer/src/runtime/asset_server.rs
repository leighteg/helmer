use crate::animation::{
    AnimationChannel, AnimationClip, AnimationLibrary, Interpolation as AnimInterp, Joint,
    Keyframe, Skeleton, Skin,
};
use crate::audio::{AudioClip, AudioLoadMode};
#[cfg(target_arch = "wasm32")]
use crate::graphics::common::renderer::MeshletLodData;
use crate::graphics::common::{
    meshlets::{build_meshlet_lod, meshlet_lod_size_bytes},
    renderer::{
        Aabb, AlphaMode, AssetStreamKind, AssetStreamingRequest, MeshLodPayload, RenderMessage,
        Vertex, build_mip_uploads, calc_mip_level_count, mip_level_data_size,
        render_message_payload_bytes,
    },
};
use crate::provided::components::Transform;
#[cfg(target_arch = "wasm32")]
use crate::runtime::asset_worker::{WorkerRequest, WorkerResponse, WorkerTextureFormat};
use crate::runtime::runtime::RuntimeTuning;
use base64::Engine;
#[cfg(not(target_arch = "wasm32"))]
use basis_universal::encoding::{ColorSpace, Compressor, CompressorParams};
use basis_universal::transcoding::{TranscodeParameters, Transcoder, TranscoderTextureFormat};
#[cfg(not(target_arch = "wasm32"))]
use basis_universal::{BasisTextureFormat, UASTC_QUALITY_MIN};
#[cfg(target_arch = "wasm32")]
use bytemuck::{Zeroable, cast_slice, cast_slice_mut};
use crossbeam_channel::{Sender, TryRecvError, TrySendError, bounded};
use glam::Quat;
use glam::{Mat4, Vec3};
use gltf::Texture as GltfTexture;
use gltf::animation::Interpolation as GltfInterpolation;
use gltf::animation::util::ReadOutputs;
use hashbrown::{HashMap, HashSet};
#[cfg(target_arch = "wasm32")]
use js_sys::{Reflect, Uint8Array};
use ktx2::Reader;
#[cfg(not(target_arch = "wasm32"))]
use memmap2::MmapOptions;
#[cfg(all(unix, not(target_arch = "wasm32")))]
use memmap2::UncheckedAdvice;
use meshopt::{
    SimplifyOptions, VertexDataAdapter, optimize_vertex_cache, optimize_vertex_cache_fifo,
    simplify, simplify_sloppy,
};
#[cfg(target_arch = "wasm32")]
use parking_lot::Mutex;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::sync::atomic::AtomicU8;
use std::{
    borrow::Cow,
    collections::VecDeque,
    marker::PhantomData,
    ops::Deref,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
    },
    time::Duration,
};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    fs::File,
    thread::{self, JoinHandle},
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::closure::Closure;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys::{
    FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetDirectoryOptions,
    FileSystemGetFileOptions, FileSystemWritableFileStream, Response, WorkerGlobalScope,
};
#[cfg(all(target_arch = "wasm32", not(feature = "asset-worker")))]
#[wasm_bindgen(module = "/worker_bridge.js")]
extern "C" {
    fn helmer_worker_init(worker_count: u32) -> bool;
    fn helmer_worker_enqueue(payload: &[u8]) -> bool;
    fn helmer_worker_enqueue_on_worker(worker_index: u32, payload: &[u8]) -> bool;
    fn helmer_worker_set_opfs_enabled(enabled: bool);
    fn helmer_worker_store_virtual_asset(path: &str, bytes: &[u8]);
    fn helmer_worker_release_scene_buffers(scene_id: u32);
    fn helmer_worker_register_callback(callback: &js_sys::Function) -> bool;
}

#[cfg(all(target_arch = "wasm32", feature = "asset-worker"))]
fn helmer_worker_init(_worker_count: u32) -> bool {
    false
}

#[cfg(all(target_arch = "wasm32", feature = "asset-worker"))]
fn helmer_worker_enqueue(_payload: &[u8]) -> bool {
    false
}

#[cfg(all(target_arch = "wasm32", feature = "asset-worker"))]
fn helmer_worker_enqueue_on_worker(_worker_index: u32, _payload: &[u8]) -> bool {
    false
}

#[cfg(all(target_arch = "wasm32", feature = "asset-worker"))]
fn helmer_worker_set_opfs_enabled(_enabled: bool) {}

#[cfg(all(target_arch = "wasm32", feature = "asset-worker"))]
fn helmer_worker_store_virtual_asset(_path: &str, _bytes: &[u8]) {}

#[cfg(all(target_arch = "wasm32", feature = "asset-worker"))]
fn helmer_worker_release_scene_buffers(_scene_id: u32) {}

#[cfg(all(target_arch = "wasm32", feature = "asset-worker"))]
fn helmer_worker_register_callback(_callback: &js_sys::Function) -> bool {
    false
}
use tracing::{info, warn};
use web_time::Instant;

const FORWARD_TA_TARGET_RES: u32 = 512;
#[cfg(target_arch = "wasm32")]
const WEB_ASSET_CREATION_LIMIT_PER_FRAME: usize = 1;
#[cfg(target_arch = "wasm32")]
const WEB_STREAMING_UPLOAD_LIMIT_PER_FRAME: usize = 4;
#[cfg(target_arch = "wasm32")]
const WEB_MAX_INFLIGHT_PER_WORKER: usize = 1;
#[cfg(target_arch = "wasm32")]
const WEB_WORKER_MAX_COUNT: usize = 4;
#[cfg(target_arch = "wasm32")]
const WEB_STREAMING_ATTEMPT_MULTIPLIER: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

fn strip_uri_query_and_fragment(uri: &str) -> &str {
    let no_fragment = uri.split_once('#').map_or(uri, |(path, _)| path);
    no_fragment
        .split_once('?')
        .map_or(no_fragment, |(path, _)| path)
}

fn percent_decode_lossy(input: &str) -> Vec<u8> {
    let bytes = input.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = bytes[i + 1];
            let lo = bytes[i + 2];
            let hi = (hi as char).to_digit(16);
            let lo = (lo as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                decoded.push(((hi << 4) | lo) as u8);
                i += 3;
                continue;
            }
        }
        decoded.push(bytes[i]);
        i += 1;
    }

    decoded
}

fn normalize_local_uri_reference(uri: &str) -> String {
    let cleaned = strip_uri_query_and_fragment(uri);
    String::from_utf8_lossy(&percent_decode_lossy(cleaned)).into_owned()
}

#[cfg(target_arch = "wasm32")]
fn path_looks_like_url(path: &Path) -> bool {
    let as_str = path.to_string_lossy();
    as_str.contains("://") || as_str.starts_with("blob:")
}

fn uri_is_remote(uri: &str) -> bool {
    uri.starts_with("http://") || uri.starts_with("https://") || uri.starts_with("blob:")
}

#[cfg(target_arch = "wasm32")]
fn normalize_uri_for_external_read(uri: &str, base_path: Option<&Path>) -> String {
    let treat_as_url = uri_is_remote(uri) || base_path.map(path_looks_like_url).unwrap_or(false);
    if treat_as_url {
        uri.to_string()
    } else {
        normalize_local_uri_reference(uri)
    }
}

fn extension_hint_from_uri(uri: &str) -> Option<String> {
    let cleaned = strip_uri_query_and_fragment(uri);
    let file_name = cleaned.rsplit('/').next().unwrap_or(cleaned);
    let ext = file_name.rsplit('.').next()?;
    if ext.is_empty() || ext.len() == file_name.len() {
        return None;
    }
    Some(ext.trim_start_matches('.').to_ascii_lowercase())
}

fn extension_hint_from_path(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| ext.trim_start_matches('.').to_ascii_lowercase())
}

fn normalize_mime_hint(mime_hint: &str) -> String {
    mime_hint
        .split(';')
        .next()
        .unwrap_or(mime_hint)
        .trim()
        .to_ascii_lowercase()
}

fn image_format_from_extension(ext_hint: &str) -> Option<image::ImageFormat> {
    match ext_hint
        .trim_start_matches('.')
        .to_ascii_lowercase()
        .as_str()
    {
        "png" | "apng" => Some(image::ImageFormat::Png),
        "jpg" | "jpeg" | "jpe" => Some(image::ImageFormat::Jpeg),
        "webp" => Some(image::ImageFormat::WebP),
        "gif" => Some(image::ImageFormat::Gif),
        "bmp" => Some(image::ImageFormat::Bmp),
        "tga" => Some(image::ImageFormat::Tga),
        "tif" | "tiff" => Some(image::ImageFormat::Tiff),
        "hdr" => Some(image::ImageFormat::Hdr),
        "exr" => Some(image::ImageFormat::OpenExr),
        "dds" => Some(image::ImageFormat::Dds),
        "pnm" | "pbm" | "pgm" | "ppm" | "pam" => Some(image::ImageFormat::Pnm),
        "ico" => Some(image::ImageFormat::Ico),
        "ff" | "farbfeld" => Some(image::ImageFormat::Farbfeld),
        "avif" => Some(image::ImageFormat::Avif),
        "qoi" => Some(image::ImageFormat::Qoi),
        _ => None,
    }
}

fn image_format_from_mime(mime_hint: &str) -> Option<image::ImageFormat> {
    match normalize_mime_hint(mime_hint).as_str() {
        "image/png" => Some(image::ImageFormat::Png),
        "image/jpeg" | "image/jpg" | "image/pjpeg" => Some(image::ImageFormat::Jpeg),
        "image/webp" => Some(image::ImageFormat::WebP),
        "image/gif" => Some(image::ImageFormat::Gif),
        "image/bmp" | "image/x-ms-bmp" => Some(image::ImageFormat::Bmp),
        "image/tga" | "image/x-tga" | "image/x-targa" => Some(image::ImageFormat::Tga),
        "image/tiff" => Some(image::ImageFormat::Tiff),
        "image/vnd.radiance" | "image/hdr" => Some(image::ImageFormat::Hdr),
        "image/exr" | "image/x-exr" => Some(image::ImageFormat::OpenExr),
        "image/dds" | "image/x-dds" => Some(image::ImageFormat::Dds),
        "image/x-portable-anymap" => Some(image::ImageFormat::Pnm),
        "image/vnd.microsoft.icon" | "image/x-icon" => Some(image::ImageFormat::Ico),
        "image/avif" => Some(image::ImageFormat::Avif),
        "image/x-qoi" => Some(image::ImageFormat::Qoi),
        _ => None,
    }
}

fn texture_image_format_hint(
    mime_hint: Option<&str>,
    ext_hint: Option<&str>,
) -> Option<image::ImageFormat> {
    mime_hint
        .and_then(image_format_from_mime)
        .or_else(|| ext_hint.and_then(image_format_from_extension))
}

pub(crate) fn decode_texture_file_bytes(
    kind: AssetKind,
    bytes: &[u8],
    path: &Path,
) -> Result<(Vec<u8>, wgpu::TextureFormat, (u32, u32)), String> {
    let ext_hint = extension_hint_from_path(path);
    decode_texture_bytes(kind, bytes, None, ext_hint.as_deref())
}

#[cfg(not(target_arch = "wasm32"))]
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
                let ext_hint = extension_hint_from_uri(uri);
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
            if uri_is_remote(uri) {
                warn!(
                    "Skipping remote image URI '{}' on native target; only local file URIs are supported",
                    uri
                );
                return Ok(None);
            }
            let image_uri = normalize_local_uri_reference(uri);
            let img_path = base.join(&image_uri);
            match std::fs::read(&img_path) {
                Ok(bytes) => {
                    let ext_hint = extension_hint_from_uri(uri)
                        .or_else(|| extension_hint_from_path(&img_path));
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
    let normalized_mime = mime_hint.map(normalize_mime_hint);
    let normalized_ext = ext_hint.map(|ext| ext.trim_start_matches('.').to_ascii_lowercase());

    let looks_like_ktx2 = matches!(normalized_mime.as_deref(), Some("image/ktx2"))
        || matches!(normalized_ext.as_deref(), Some("ktx2"))
        || is_ktx2_bytes(bytes);
    if looks_like_ktx2 {
        return decode_ktx2(bytes, kind);
    }

    let decoded = if let Some(format_hint) =
        texture_image_format_hint(normalized_mime.as_deref(), normalized_ext.as_deref())
    {
        image::load_from_memory_with_format(bytes, format_hint)
            .or_else(|_| image::load_from_memory(bytes))
    } else {
        image::load_from_memory(bytes)
    }
    .map_err(|e| e.to_string())?;
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

#[cfg(not(target_arch = "wasm32"))]
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

    let decoded = if is_base64 {
        base64::engine::general_purpose::STANDARD
            .decode(data_part)
            .map_err(|e| format!("Failed to decode data URI: {}", e))?
    } else {
        percent_decode_lossy(data_part)
    };
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
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: Option<f32>,
}

/// Represents a single drawable element in a scene graph, corresponding to a glTF primitive.
#[derive(Debug, Clone)]
pub struct SceneNode {
    pub mesh: Handle<Mesh>,
    pub material: Handle<Material>,
    pub transform: glam::Mat4,
    pub skin_index: Option<usize>,
    pub node_index: usize,
}

/// The final representation of a scene, stored in the AssetServer.
#[derive(Debug, Clone)]
pub struct Scene {
    pub nodes: Vec<SceneNode>,
    pub skins: Arc<RwLock<Vec<Arc<Skin>>>>,
    pub animations: Arc<RwLock<Vec<Arc<AnimationLibrary>>>>,
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
pub(crate) struct LowResTexture {
    pub data: Arc<[u8]>,
    pub format: wgpu::TextureFormat,
    pub dimensions: (u32, u32),
}

pub(crate) fn generate_low_res_from_parts(
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
pub(crate) struct ParsedGltfScene {
    pub(crate) doc: Arc<gltf::Document>,
    pub(crate) buffers: Option<Arc<Vec<StreamedBuffer>>>,
    pub(crate) buffers_bytes: usize,
    pub(crate) base_path: Option<PathBuf>,
    pub(crate) scene_path: Option<PathBuf>,
    pub(crate) textures: Vec<TextureRequest>,
    pub(crate) materials: Vec<IntermediateMaterial>,
    pub(crate) mesh_primitives: Vec<MeshPrimitiveDesc>,
    pub(crate) nodes: Vec<GltfNodeDesc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct MeshPrimitiveDesc {
    pub(crate) mesh_index: usize,
    pub(crate) primitive_index: usize,
    pub(crate) material_index: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GltfNodeDesc {
    pub(crate) primitive_desc_index: usize,
    pub(crate) material_index: usize,
    pub(crate) transform: glam::Mat4,
    pub(crate) node_index: usize,
    pub(crate) skin_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct TextureRequest {
    tex_index: usize,
    kind: AssetKind,
}

/// An intermediate material representation that uses texture indices
/// instead of string paths, suitable for parsed scene data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IntermediateMaterial {
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
    alpha_mode: AlphaMode,
    alpha_cutoff: Option<f32>,
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
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
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
    #[serde(default)]
    pub alpha_mode: AlphaMode,
    #[serde(default)]
    pub alpha_cutoff: Option<f32>,
}

pub(crate) trait BufferSource: Send + Sync {
    fn as_slice(&self) -> &[u8];
}

impl BufferSource for gltf::buffer::Data {
    fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

#[derive(Debug)]
pub(crate) struct StreamedBuffer {
    backing: StreamedBufferBacking,
}

#[derive(Debug)]
enum StreamedBufferBacking {
    Owned(Vec<u8>),
    #[cfg(not(target_arch = "wasm32"))]
    Mapped(memmap2::Mmap),
}

impl StreamedBuffer {
    pub(crate) fn owned(data: Vec<u8>) -> Self {
        Self {
            backing: StreamedBufferBacking::Owned(data),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn mapped(map: memmap2::Mmap) -> Self {
        Self {
            backing: StreamedBufferBacking::Mapped(map),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.deref().len()
    }

    #[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(target_arch = "wasm32")]
    fn release_pages(&self) {}
}

impl Deref for StreamedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match &self.backing {
            StreamedBufferBacking::Owned(data) => data.as_slice(),
            #[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
enum SceneContextStatus {
    Ready {
        doc: Arc<gltf::Document>,
        buffers: SceneBufferGuard,
        base_path: Option<PathBuf>,
    },
    Pending,
    Missing,
}

#[cfg(target_arch = "wasm32")]
enum SceneContextStatus {
    Ready,
    Pending,
    Missing,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
pub(crate) struct WebAssetIo {
    opfs_enabled: Arc<AtomicBool>,
    virtual_assets: Arc<RwLock<HashMap<String, Arc<Vec<u8>>>>>,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_OPFS_ROOT: RefCell<Option<FileSystemDirectoryHandle>> = RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
enum WebGlobalScope {
    Window(web_sys::Window),
    Worker(WorkerGlobalScope),
}

#[cfg(target_arch = "wasm32")]
impl WebGlobalScope {
    fn current() -> Result<Self, String> {
        if let Some(window) = web_sys::window() {
            return Ok(Self::Window(window));
        }
        let global = js_sys::global();
        global
            .dyn_into::<WorkerGlobalScope>()
            .map(Self::Worker)
            .map_err(|_| "Missing window/worker global scope".to_string())
    }

    fn fetch_with_str(&self, url: &str) -> js_sys::Promise {
        match self {
            Self::Window(window) => window.fetch_with_str(url),
            Self::Worker(scope) => scope.fetch_with_str(url),
        }
    }

    fn storage_manager(&self) -> web_sys::StorageManager {
        match self {
            Self::Window(window) => window.navigator().storage(),
            Self::Worker(scope) => scope.navigator().storage(),
        }
    }

    fn hardware_concurrency(&self) -> Option<u32> {
        let count = match self {
            Self::Window(window) => window.navigator().hardware_concurrency(),
            Self::Worker(scope) => scope.navigator().hardware_concurrency(),
        };
        if count > 0.0 {
            Some(count as u32)
        } else {
            None
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl WebAssetIo {
    pub(crate) fn new() -> Self {
        Self {
            opfs_enabled: Arc::new(AtomicBool::new(true)),
            virtual_assets: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub(crate) fn set_opfs_enabled(&self, enabled: bool) {
        self.opfs_enabled.store(enabled, AtomicOrdering::Relaxed);
    }

    fn normalize_path(path: &Path) -> String {
        let mut key = path.to_string_lossy().replace('\\', "/");
        while key.starts_with("./") {
            key = key.trim_start_matches("./").to_string();
        }
        key.trim_start_matches('/').to_string()
    }

    fn normalize_key(key: &str) -> String {
        let mut normalized = key.replace('\\', "/");
        while normalized.starts_with("./") {
            normalized = normalized.trim_start_matches("./").to_string();
        }
        normalized.trim_start_matches('/').to_string()
    }

    fn resolve_uri(base: Option<&Path>, uri: &str) -> String {
        if uri.starts_with("data:")
            || uri.starts_with("http://")
            || uri.starts_with("https://")
            || uri.starts_with("blob:")
        {
            return uri.to_string();
        }

        let trimmed = uri.trim_start_matches("./");
        if let Some(base) = base {
            let base_key = Self::normalize_path(base);
            if base_key.is_empty() {
                trimmed.to_string()
            } else {
                format!("{}/{}", base_key.trim_end_matches('/'), trimmed)
            }
        } else {
            trimmed.to_string()
        }
    }

    pub(crate) async fn read_path(&self, path: &Path) -> Result<Vec<u8>, String> {
        let key = Self::normalize_path(path);
        self.read_key(&key).await
    }

    pub(crate) async fn read_uri(&self, base: Option<&Path>, uri: &str) -> Result<Vec<u8>, String> {
        let key = Self::resolve_uri(base, uri);
        self.read_key(&key).await
    }

    fn find_virtual_asset(assets: &HashMap<String, Arc<Vec<u8>>>, key: &str) -> Option<Vec<u8>> {
        if let Some(bytes) = assets.get(key) {
            return Some(bytes.as_ref().clone());
        }
        let mut start = 0usize;
        while let Some(pos) = key[start..].find('/') {
            start += pos + 1;
            if start >= key.len() {
                break;
            }
            let slice = &key[start..];
            if let Some(bytes) = assets.get(slice) {
                return Some(bytes.as_ref().clone());
            }
        }
        None
    }

    async fn read_key(&self, key: &str) -> Result<Vec<u8>, String> {
        let normalized = Self::normalize_key(key);
        if let Some(bytes) = {
            let virtual_assets = self.virtual_assets.read();
            Self::find_virtual_asset(&virtual_assets, &normalized)
        } {
            return Ok(bytes);
        }
        if normalized.starts_with("http://")
            || normalized.starts_with("https://")
            || normalized.starts_with("blob:")
        {
            return self.fetch_bytes(&normalized).await;
        }
        let opfs_enabled = self.opfs_enabled.load(AtomicOrdering::Relaxed);
        if opfs_enabled {
            if let Some(bytes) = self.try_read_opfs(&normalized).await? {
                return Ok(bytes);
            }
        }

        let bytes = self.fetch_bytes(&normalized).await?;
        if opfs_enabled {
            let _ = self.write_opfs(&normalized, &bytes).await;
        }
        Ok(bytes)
    }

    pub(crate) fn store_virtual_asset_path(&self, path: &Path, bytes: Vec<u8>) {
        let key = Self::normalize_path(path);
        let shared = Arc::new(bytes);
        let mut assets = self.virtual_assets.write();
        assets.insert(key.clone(), Arc::clone(&shared));

        let mut relative = key.as_str();
        if let Some(scheme_idx) = relative.find("://") {
            let after_scheme = &relative[scheme_idx + 3..];
            relative = match after_scheme.split_once('/') {
                Some((_, path)) => path,
                None => "",
            };
        }

        if relative.is_empty() {
            return;
        }

        let mut insert_alias = |alias: &str| {
            if let Some(existing) = assets.get(alias) {
                if !Arc::ptr_eq(existing, &shared) {
                    warn!(
                        "Virtual asset alias '{}' already mapped; keeping existing entry",
                        alias
                    );
                }
                return;
            }
            assets.insert(alias.to_string(), Arc::clone(&shared));
        };

        if relative != key {
            insert_alias(relative);
        }

        let segments: Vec<&str> = relative
            .split('/')
            .filter(|segment| !segment.is_empty())
            .collect();
        if segments.len() > 1 {
            for start in 1..segments.len() {
                let alias = segments[start..].join("/");
                if !alias.is_empty() {
                    insert_alias(&alias);
                }
            }
        }
    }

    async fn opfs_root(&self) -> Option<FileSystemDirectoryHandle> {
        if !self.opfs_enabled.load(AtomicOrdering::Relaxed) {
            return None;
        }
        if let Some(root) = WEB_OPFS_ROOT.with(|slot| slot.borrow().clone()) {
            return Some(root);
        }
        let scope = WebGlobalScope::current().ok()?;
        let storage = scope.storage_manager();
        let has_get_directory = Reflect::get(storage.as_ref(), &JsValue::from_str("getDirectory"))
            .ok()
            .map(|value| value.is_function())
            .unwrap_or(false);
        if !has_get_directory {
            warn!("OPFS getDirectory unavailable; disabling OPFS cache.");
            self.opfs_enabled.store(false, AtomicOrdering::Relaxed);
            return None;
        }
        let handle = JsFuture::from(storage.get_directory()).await.ok()?;
        let root: FileSystemDirectoryHandle = handle.dyn_into().ok()?;
        WEB_OPFS_ROOT.with(|slot| {
            *slot.borrow_mut() = Some(root.clone());
        });
        Some(root)
    }

    async fn opfs_open_file(
        &self,
        key: &str,
        create: bool,
    ) -> Result<Option<FileSystemFileHandle>, String> {
        let Some(mut current) = self.opfs_root().await else {
            return Ok(None);
        };
        let parts: Vec<&str> = key.split('/').filter(|p| !p.is_empty()).collect();
        if parts.is_empty() {
            return Ok(None);
        }

        for dir in &parts[..parts.len().saturating_sub(1)] {
            let mut opts = FileSystemGetDirectoryOptions::new();
            opts.set_create(create);
            let promise = current.get_directory_handle_with_options(dir, &opts);
            let value = match JsFuture::from(promise).await {
                Ok(val) => val,
                Err(_) => return Ok(None),
            };
            current = value
                .dyn_into::<FileSystemDirectoryHandle>()
                .map_err(|_| "Invalid directory handle".to_string())?;
        }

        let mut opts = FileSystemGetFileOptions::new();
        opts.set_create(create);
        let promise = current.get_file_handle_with_options(parts[parts.len() - 1], &opts);
        let value = match JsFuture::from(promise).await {
            Ok(val) => val,
            Err(_) => return Ok(None),
        };
        let handle = value
            .dyn_into::<FileSystemFileHandle>()
            .map_err(|_| "Invalid file handle".to_string())?;
        Ok(Some(handle))
    }

    async fn try_read_opfs(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        let Some(handle) = self.opfs_open_file(key, false).await? else {
            return Ok(None);
        };
        let file = JsFuture::from(handle.get_file())
            .await
            .map_err(js_err_to_string)?;
        let file: web_sys::File = file.dyn_into().map_err(|_| "Invalid file".to_string())?;
        let buffer = JsFuture::from(file.array_buffer())
            .await
            .map_err(js_err_to_string)?;
        let array = Uint8Array::new(&buffer);
        let mut bytes = vec![0u8; array.length() as usize];
        array.copy_to(&mut bytes);
        Ok(Some(bytes))
    }

    async fn write_opfs(&self, key: &str, bytes: &[u8]) -> Result<(), String> {
        let Some(handle) = self.opfs_open_file(key, true).await? else {
            return Ok(());
        };
        let stream = JsFuture::from(handle.create_writable())
            .await
            .map_err(js_err_to_string)?;
        let stream: FileSystemWritableFileStream = stream
            .dyn_into()
            .map_err(|_| "Invalid writable stream".to_string())?;
        let promise = stream
            .write_with_u8_array(bytes)
            .map_err(js_err_to_string)?;
        let _ = JsFuture::from(promise).await.map_err(js_err_to_string)?;
        let _ = JsFuture::from(stream.close())
            .await
            .map_err(js_err_to_string)?;
        Ok(())
    }

    async fn fetch_bytes(&self, url: &str) -> Result<Vec<u8>, String> {
        let scope = WebGlobalScope::current().map_err(|err| err.to_string())?;
        let resp_value = JsFuture::from(scope.fetch_with_str(url))
            .await
            .map_err(js_err_to_string)?;
        let resp: Response = resp_value
            .dyn_into()
            .map_err(|_| "Invalid fetch response".to_string())?;
        if !resp.ok() {
            return Err(format!("HTTP {} while fetching {}", resp.status(), url));
        }
        let buffer = JsFuture::from(resp.array_buffer().map_err(js_err_to_string)?)
            .await
            .map_err(js_err_to_string)?;
        let array = Uint8Array::new(&buffer);
        let mut bytes = vec![0u8; array.length() as usize];
        array.copy_to(&mut bytes);
        Ok(bytes)
    }
}

#[cfg(target_arch = "wasm32")]
fn js_err_to_string(err: impl std::fmt::Debug) -> String {
    format!("{:?}", err)
}

#[cfg(target_arch = "wasm32")]
fn web_worker_thread_count(worker_queue_capacity: usize) -> usize {
    let override_count = web_worker_override_count();
    let concurrency = WebGlobalScope::current()
        .ok()
        .and_then(|scope| scope.hardware_concurrency())
        .unwrap_or(4) as usize;
    let mut threads = concurrency.saturating_sub(1).max(1);
    if let Some(override_count) = override_count {
        threads = override_count.max(1);
    } else {
        threads = threads.min(WEB_WORKER_MAX_COUNT.max(1));
    }
    if worker_queue_capacity > 0 {
        threads = threads.min(worker_queue_capacity);
    }
    threads
}

#[cfg(target_arch = "wasm32")]
fn web_worker_override_count() -> Option<usize> {
    let global = js_sys::global();
    let value = Reflect::get(&global, &JsValue::from_str("HELMER_WEB_WORKERS")).ok()?;
    let count = value.as_f64()?;
    if count.is_finite() && count >= 1.0 {
        Some(count.floor() as usize)
    } else {
        None
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct WebWorkerBridge {
    status: Arc<AtomicU8>,
    worker_count: Arc<AtomicUsize>,
}

#[cfg(target_arch = "wasm32")]
impl WebWorkerBridge {
    const UNINITIALIZED: u8 = 0;
    const INITIALIZING: u8 = 1;
    const READY: u8 = 2;
    const FAILED: u8 = 3;

    fn new() -> Self {
        Self {
            status: Arc::new(AtomicU8::new(Self::UNINITIALIZED)),
            worker_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn ensure_initialized(&self, worker_queue_capacity: usize) {
        if self
            .status
            .compare_exchange(
                Self::UNINITIALIZED,
                Self::INITIALIZING,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            )
            .is_err()
        {
            return;
        }
        let threads = web_worker_thread_count(worker_queue_capacity).max(1);
        let ok = helmer_worker_init(threads as u32);
        if ok {
            self.status.store(Self::READY, AtomicOrdering::Relaxed);
            self.worker_count
                .store(threads.max(1), AtomicOrdering::Relaxed);
            info!(
                "Web asset worker pool initialized with {} workers.",
                threads
            );
        } else {
            warn!("Web asset worker pool init failed; asset loading will stall.");
            self.status.store(Self::FAILED, AtomicOrdering::Relaxed);
            self.worker_count.store(0, AtomicOrdering::Relaxed);
        }
    }

    fn is_ready(&self) -> bool {
        self.status.load(AtomicOrdering::Relaxed) == Self::READY
    }

    fn is_failed(&self) -> bool {
        self.status.load(AtomicOrdering::Relaxed) == Self::FAILED
    }

    fn worker_count(&self) -> usize {
        self.worker_count.load(AtomicOrdering::Relaxed)
    }

    fn mark_failed(&self) {
        self.status.store(Self::FAILED, AtomicOrdering::Relaxed);
        self.worker_count.store(0, AtomicOrdering::Relaxed);
    }
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
    Audio {
        id: usize,
        path: PathBuf,
        mode: AudioLoadMode,
    },
    Scene {
        id: usize,
        path: PathBuf,
    },
    SceneBuffers {
        scene_id: usize,
        #[cfg(not(target_arch = "wasm32"))]
        doc: Arc<gltf::Document>,
        #[cfg(not(target_arch = "wasm32"))]
        base_path: Option<PathBuf>,
        #[cfg(not(target_arch = "wasm32"))]
        scene_path: Option<PathBuf>,
    },
    StreamMesh {
        id: usize,
        scene_id: usize,
        desc: MeshPrimitiveDesc,
        #[cfg(not(target_arch = "wasm32"))]
        doc: Arc<gltf::Document>,
        #[cfg(not(target_arch = "wasm32"))]
        buffers: SceneBufferGuard,
        tuning: AssetStreamingTuning,
    },
    StreamTexture {
        id: usize,
        scene_id: usize,
        tex_index: usize,
        kind: AssetKind,
        #[cfg(not(target_arch = "wasm32"))]
        doc: Arc<gltf::Document>,
        #[cfg(not(target_arch = "wasm32"))]
        buffers: SceneBufferGuard,
        #[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
type SceneLoadData = ParsedGltfScene;
#[cfg(target_arch = "wasm32")]
type SceneLoadData = crate::runtime::asset_worker::WorkerSceneSummary;

#[derive(Debug)]
struct SceneBuffersPayload {
    #[cfg(not(target_arch = "wasm32"))]
    buffers: Vec<StreamedBuffer>,
    buffers_bytes: usize,
}

#[cfg(target_arch = "wasm32")]
struct WebWorkerDispatch {
    request: WorkerRequest,
    scene_affinity: Option<usize>,
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
    Audio {
        id: usize,
        clip: AudioClip,
    },
    Scene {
        id: usize,
        data: SceneLoadData,
    },
    SceneBuffers {
        scene_id: usize,
        payload: SceneBuffersPayload,
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
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

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct SceneAssetContext {
    doc: Arc<gltf::Document>,
    buffers: Option<Arc<Vec<StreamedBuffer>>>,
    buffers_bytes: usize,
    buffers_last_used: Instant,
    base_path: Option<PathBuf>,
    scene_path: Option<PathBuf>,
    pending_assets: usize,
    skins_loaded: bool,
    animations_loaded: bool,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct SceneAssetContext {
    buffers_ready: bool,
    buffers_bytes: usize,
    buffers_last_used: Instant,
    base_path: Option<PathBuf>,
    scene_path: Option<PathBuf>,
    pending_assets: usize,
    skins_loaded: bool,
    animations_loaded: bool,
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

#[derive(Clone)]
struct AudioSource {
    path: PathBuf,
    mode: AudioLoadMode,
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
    asset_base_path: RwLock<Option<String>>,
    #[cfg(not(target_arch = "wasm32"))]
    _worker_handles: Vec<JoinHandle<()>>,
    pending_worker_requests: RwLock<VecDeque<AssetLoadRequest>>,

    stream_request_receiver: crossbeam_channel::Receiver<AssetStreamingRequest>,
    streaming_backlog: RwLock<VecDeque<AssetStreamingRequest>>,
    reupload_queue: RwLock<VecDeque<AssetStreamingRequest>>,
    streaming_inflight: RwLock<HashMap<(AssetStreamKind, usize), AssetStreamingRequest>>,
    latest_streaming_plan: RwLock<HashMap<(AssetStreamKind, usize), f32>>,
    streaming_plan_epoch: AtomicU64,
    streaming_backlog_epoch: AtomicU64,
    streaming_backlog_dirty: AtomicBool,
    scene_contexts: RwLock<HashMap<usize, SceneAssetContext>>,
    scene_buffer_bytes: AtomicUsize,
    scene_buffer_inflight: RwLock<HashSet<usize>>,
    texture_sources: RwLock<HashMap<usize, TextureSource>>,
    mesh_sources: RwLock<HashMap<usize, MeshSource>>,
    material_sources: RwLock<HashMap<usize, MaterialSource>>,
    audio_sources: RwLock<HashMap<usize, AudioSource>>,
    mesh_cache: RwLock<HashMap<usize, Arc<Mesh>>>,
    texture_cache: RwLock<HashMap<usize, Arc<CachedTexture>>>,
    material_cache: RwLock<HashMap<usize, Arc<MaterialGpuData>>>,
    audio_cache: RwLock<HashMap<usize, Arc<AudioClip>>>,
    mesh_meta: RwLock<HashMap<usize, CacheEntryMeta>>,
    texture_meta: RwLock<HashMap<usize, CacheEntryMeta>>,
    material_meta: RwLock<HashMap<usize, CacheEntryMeta>>,
    audio_meta: RwLock<HashMap<usize, CacheEntryMeta>>,
    mesh_budget: usize,
    texture_budget: usize,
    material_budget: usize,
    audio_budget: usize,
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
    #[cfg(target_arch = "wasm32")]
    web_io: WebAssetIo,
    #[cfg(target_arch = "wasm32")]
    web_worker_bridge: WebWorkerBridge,
    #[cfg(target_arch = "wasm32")]
    web_pending_responses: Mutex<VecDeque<Vec<u8>>>,
    #[cfg(target_arch = "wasm32")]
    inflight_worker_requests: Arc<AtomicUsize>,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_ASSET_SERVER: RefCell<Option<Arc<Mutex<AssetServer>>>> = RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_WORKER_RESPONSE_CB: RefCell<Option<Closure<dyn FnMut(JsValue)>>> =
        RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
pub fn set_web_asset_server(server: Arc<Mutex<AssetServer>>) {
    WEB_ASSET_SERVER.with(|slot| {
        *slot.borrow_mut() = Some(server);
    });
    #[cfg(not(feature = "asset-worker"))]
    register_worker_response_callback();
}

#[cfg(all(target_arch = "wasm32", not(feature = "asset-worker")))]
fn register_worker_response_callback() {
    WEB_WORKER_RESPONSE_CB.with(|slot| {
        if slot.borrow().is_some() {
            return;
        }
        let closure = Closure::wrap(Box::new(move |value: JsValue| {
            let array = Uint8Array::new(&value);
            let mut bytes = vec![0u8; array.length() as usize];
            array.copy_to(&mut bytes);
            helmer_worker_response(bytes);
        }) as Box<dyn FnMut(JsValue)>);
        let func: &js_sys::Function = closure.as_ref().unchecked_ref();
        let _ = helmer_worker_register_callback(func);
        *slot.borrow_mut() = Some(closure);
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn helmer_store_virtual_asset(path: String, bytes: Vec<u8>) {
    WEB_ASSET_SERVER.with(|slot| {
        if let Some(server) = slot.borrow().as_ref() {
            server.lock().store_virtual_asset(path, bytes);
        } else {
            warn!(
                "Web asset server not initialized; dropping virtual asset '{}'",
                path
            );
        }
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn helmer_worker_response(bytes: Vec<u8>) {
    WEB_ASSET_SERVER.with(|slot| {
        let binding = slot.borrow();
        let Some(server) = binding.as_ref() else {
            warn!("Web asset server not initialized; dropping worker response.");
            return;
        };
        let guard = server.lock();
        guard.web_pending_responses.lock().push_back(bytes);
    });
}

#[cfg(target_arch = "wasm32")]
fn worker_response_to_result(response: WorkerResponse) -> Option<AssetLoadResult> {
    match response {
        WorkerResponse::Mesh {
            id,
            scene_id,
            payload,
        } => match mesh_payload_from_worker(payload) {
            Some((lods, total_lods, bounds)) => Some(AssetLoadResult::Mesh {
                id,
                scene_id,
                lods,
                total_lods,
                bounds,
            }),
            None => {
                if let Some(scene_id) = scene_id {
                    Some(AssetLoadResult::StreamFailure {
                        kind: AssetStreamKind::Mesh,
                        id,
                        scene_id,
                    })
                } else {
                    warn!(
                        "Dropped worker mesh payload for asset {} (decode failed).",
                        id
                    );
                    None
                }
            }
        },
        WorkerResponse::Texture {
            id,
            scene_id,
            name,
            kind,
            data,
        } => Some(AssetLoadResult::Texture {
            id,
            scene_id,
            name,
            kind,
            data: (data.data, data.format.to_wgpu(), data.dimensions),
        }),
        WorkerResponse::Material { id, data } => Some(AssetLoadResult::Material { id, data }),
        WorkerResponse::Scene { id, summary } => Some(AssetLoadResult::Scene { id, data: summary }),
        WorkerResponse::SceneBuffers {
            scene_id,
            buffers_bytes,
        } => Some(AssetLoadResult::SceneBuffers {
            scene_id,
            payload: SceneBuffersPayload { buffers_bytes },
        }),
        WorkerResponse::SceneBuffersFailed { scene_id, error } => {
            warn!("Scene buffer load failed for {}: {}", scene_id, error);
            Some(AssetLoadResult::SceneBuffersFailed { scene_id })
        }
        WorkerResponse::LowResTexture {
            id,
            name,
            kind,
            data,
        } => {
            let low = LowResTexture {
                data: Arc::from(data.data),
                format: data.format.to_wgpu(),
                dimensions: data.dimensions,
            };
            Some(AssetLoadResult::LowResTexture {
                id,
                name,
                kind,
                data: low,
            })
        }
        WorkerResponse::StreamFailure { kind, id, scene_id } => {
            Some(AssetLoadResult::StreamFailure { kind, id, scene_id })
        }
        WorkerResponse::Error { message } => {
            warn!("Asset worker error: {}", message);
            None
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn mesh_payload_from_worker(
    payload: crate::runtime::asset_worker::WorkerMeshPayload,
) -> Option<(Vec<MeshLodPayload>, usize, Aabb)> {
    let bounds = Aabb {
        min: Vec3::from(payload.bounds.min),
        max: Vec3::from(payload.bounds.max),
    };
    let mut lods = Vec::with_capacity(payload.lods.len());
    for lod in payload.lods {
        let Some(vertices) = vertices_from_worker_bytes(&lod.vertices) else {
            return None;
        };
        let indices: Vec<u32> = lod.indices;
        let meshlets = MeshletLodData {
            descs: Arc::from(lod.meshlets.descs),
            vertices: Arc::from(lod.meshlets.vertices),
            indices: Arc::from(lod.meshlets.indices),
        };
        lods.push(MeshLodPayload {
            lod_index: lod.lod_index,
            vertices: Arc::from(vertices),
            indices: Arc::from(indices),
            meshlets,
        });
    }
    Some((lods, payload.total_lods, bounds))
}

#[cfg(target_arch = "wasm32")]
fn vertices_from_worker_bytes(bytes: &[u8]) -> Option<Vec<Vertex>> {
    let elem_size = std::mem::size_of::<Vertex>();
    if bytes.len() % elem_size != 0 {
        warn!(
            "Worker mesh payload vertex buffer size mismatch ({} bytes, expected multiple of {}).",
            bytes.len(),
            elem_size
        );
        return None;
    }
    let count = bytes.len() / elem_size;
    let mut vertices = vec![Vertex::zeroed(); count];
    cast_slice_mut(vertices.as_mut_slice()).copy_from_slice(bytes);
    Some(vertices)
}

#[derive(Debug, Clone)]
struct CacheEntryMeta {
    last_used: Instant,
    size_bytes: usize,
    priority: f32,
    plan_epoch: u64,
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
        #[cfg(target_arch = "wasm32")]
        let _ = request_receiver;
        #[cfg(target_arch = "wasm32")]
        let _ = result_sender;
        #[cfg(not(target_arch = "wasm32"))]
        let mut worker_handles = Vec::new();

        #[cfg(not(target_arch = "wasm32"))]
        {
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
                                            warn!(
                                                "Failed to parse glTF mesh '{}': {}",
                                                path_str, e
                                            );
                                            None
                                        }
                                    }
                                } else {
                                    match std::fs::read(&path) {
                                        Ok(bytes) => match parse_glb(&bytes) {
                                            Ok(data) => Some(data),
                                            Err(e) => {
                                                warn!(
                                                    "Failed to parse glb mesh '{}': {}",
                                                    path_str, e
                                                );
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
                                build_mesh_payload(vertices, indices, bounds, &tuning).map(
                                    |payload| AssetLoadResult::Mesh {
                                        id,
                                        scene_id: None,
                                        lods: payload.lods,
                                        total_lods: payload.total_lods,
                                        bounds: payload.bounds,
                                    },
                                )
                            }
                            AssetLoadRequest::Texture { id, path, kind } => load_and_parse(
                                id,
                                &path,
                                |bytes| decode_texture_file_bytes(kind, bytes, &path),
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
                            AssetLoadRequest::Audio { id, path, mode } => {
                                match AudioClip::from_path_with_mode(&path, mode) {
                                    Ok(clip) => Some(AssetLoadResult::Audio { id, clip }),
                                    Err(e) => {
                                        warn!(
                                            "Failed to load audio '{}' for handle {}: {}",
                                            path_str, id, e
                                        );
                                        None
                                    }
                                }
                            }
                            AssetLoadRequest::Scene { id, path } => {
                                if is_gltf_path(&path) {
                                    match parse_scene_from_gltf_path(&path) {
                                        Ok(data) => Some(AssetLoadResult::Scene { id, data }),
                                        Err(e) => {
                                            warn!(
                                                "Failed to parse glTF scene '{}': {}",
                                                path_str, e
                                            );
                                            None
                                        }
                                    }
                                } else {
                                    match parse_scene_glb_path(&path) {
                                        Ok(data) => Some(AssetLoadResult::Scene { id, data }),
                                        Err(e) => {
                                            warn!(
                                                "Failed to parse glTF scene '{}': {}",
                                                path_str, e
                                            );
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
                                        payload: SceneBuffersPayload {
                                            buffers,
                                            buffers_bytes,
                                        },
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
        }

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
        let audio_budget = std::env::var("HELMER_ASSET_BUDGET_AUDIO_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256)
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

        #[cfg(target_arch = "wasm32")]
        let web_worker_bridge = {
            let bridge = WebWorkerBridge::new();
            bridge.ensure_initialized(worker_queue_capacity);
            bridge
        };

        Self {
            scenes: Arc::new(RwLock::new(HashMap::new())),
            mesh_aabb_map: Arc::new(RwLock::new(MeshAabbMap::default())),
            next_id: AtomicUsize::new(0),
            request_sender,
            result_receiver,
            asset_sender,
            tuning,
            asset_base_path: RwLock::new(None),
            #[cfg(not(target_arch = "wasm32"))]
            _worker_handles: worker_handles,
            pending_worker_requests: RwLock::new(VecDeque::new()),
            stream_request_receiver,
            streaming_backlog: RwLock::new(VecDeque::new()),
            reupload_queue: RwLock::new(VecDeque::new()),
            streaming_inflight: RwLock::new(HashMap::new()),
            latest_streaming_plan: RwLock::new(HashMap::new()),
            streaming_plan_epoch: AtomicU64::new(0),
            streaming_backlog_epoch: AtomicU64::new(0),
            streaming_backlog_dirty: AtomicBool::new(false),
            scene_contexts: RwLock::new(HashMap::new()),
            scene_buffer_bytes: AtomicUsize::new(0),
            scene_buffer_inflight: RwLock::new(HashSet::new()),
            texture_sources: RwLock::new(HashMap::new()),
            mesh_sources: RwLock::new(HashMap::new()),
            material_sources: RwLock::new(HashMap::new()),
            audio_sources: RwLock::new(HashMap::new()),
            mesh_cache: RwLock::new(HashMap::new()),
            texture_cache: RwLock::new(HashMap::new()),
            material_cache: RwLock::new(HashMap::new()),
            audio_cache: RwLock::new(HashMap::new()),
            mesh_meta: RwLock::new(HashMap::new()),
            texture_meta: RwLock::new(HashMap::new()),
            material_meta: RwLock::new(HashMap::new()),
            audio_meta: RwLock::new(HashMap::new()),
            mesh_budget,
            texture_budget,
            material_budget,
            audio_budget,
            scene_buffer_budget,
            asset_streaming_tuning: AssetStreamingTuning::default(),
            asset_creation_limit_per_frame: {
                #[cfg(target_arch = "wasm32")]
                {
                    WEB_ASSET_CREATION_LIMIT_PER_FRAME
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    4
                }
            },
            streaming_upload_limit_per_frame: {
                #[cfg(target_arch = "wasm32")]
                {
                    WEB_STREAMING_UPLOAD_LIMIT_PER_FRAME
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    24
                }
            },
            streaming_backlog_limit: 16_384,
            cache_idle_ms,
            cache_eviction_limit,
            worker_queue_capacity,
            low_res_inflight: RwLock::new(HashSet::new()),
            #[cfg(target_arch = "wasm32")]
            web_io: WebAssetIo::new(),
            #[cfg(target_arch = "wasm32")]
            web_worker_bridge,
            #[cfg(target_arch = "wasm32")]
            web_pending_responses: Mutex::new(VecDeque::new()),
            #[cfg(target_arch = "wasm32")]
            inflight_worker_requests: Arc::new(AtomicUsize::new(0)),
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
        let plan_epoch = self
            .streaming_plan_epoch
            .fetch_add(1, AtomicOrdering::Relaxed)
            .wrapping_add(1);
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
                            meta.priority = *priority;
                            meta.plan_epoch = plan_epoch;
                        }
                    }
                    AssetStreamKind::Material => {
                        if let Some(meta) = material_meta.get_mut(id) {
                            meta.last_used = now;
                            meta.priority = *priority;
                            meta.plan_epoch = plan_epoch;
                        }
                    }
                    AssetStreamKind::Texture => {
                        if let Some(meta) = texture_meta.get_mut(id) {
                            meta.last_used = now;
                            meta.priority = *priority;
                            meta.plan_epoch = plan_epoch;
                        }
                    }
                }
            }
        }
        *self.latest_streaming_plan.write() = map;
        self.streaming_backlog_dirty
            .store(true, AtomicOrdering::Relaxed);
    }

    pub fn request_scene_assets(
        &self,
        handle: &Handle<Scene>,
        max_lod: Option<usize>,
        priority: f32,
    ) {
        let Some(scene) = self.get_scene(handle) else {
            return;
        };

        let mut mesh_ids: HashSet<usize> = HashSet::new();
        let mut material_ids: HashSet<usize> = HashSet::new();
        for node in &scene.nodes {
            mesh_ids.insert(node.mesh.id);
            material_ids.insert(node.material.id);
        }

        if mesh_ids.is_empty() && material_ids.is_empty() {
            return;
        }

        let mut requests = Vec::with_capacity(
            mesh_ids
                .len()
                .saturating_add(material_ids.len())
                .saturating_mul(2),
        );

        for id in mesh_ids {
            requests.push(AssetStreamingRequest {
                id,
                kind: AssetStreamKind::Mesh,
                priority,
                max_lod,
                force_low_res: false,
            });
        }

        for id in material_ids.iter().copied() {
            requests.push(AssetStreamingRequest {
                id,
                kind: AssetStreamKind::Material,
                priority,
                max_lod: None,
                force_low_res: false,
            });
        }

        let mut texture_ids: HashSet<usize> = HashSet::new();
        for mat_id in material_ids {
            if let Some(textures) = self.material_texture_ids(mat_id) {
                for tex in textures.iter().flatten() {
                    texture_ids.insert(*tex);
                }
            }
        }

        for id in texture_ids {
            requests.push(AssetStreamingRequest {
                id,
                kind: AssetStreamKind::Texture,
                priority,
                max_lod: None,
                force_low_res: false,
            });
        }

        self.publish_streaming_plan(&requests);
        self.enqueue_streaming_requests(&requests);
    }

    fn stream_kind_rank(kind: AssetStreamKind) -> u8 {
        match kind {
            AssetStreamKind::Mesh => 0,
            AssetStreamKind::Material => 1,
            AssetStreamKind::Texture => 2,
        }
    }

    fn current_plan_epoch(&self) -> u64 {
        self.streaming_plan_epoch.load(AtomicOrdering::Relaxed)
    }

    fn plan_entry(&self, kind: AssetStreamKind, id: usize) -> Option<f32> {
        self.latest_streaming_plan.read().get(&(kind, id)).copied()
    }

    fn enqueue_streaming_requests(&self, requests: &[AssetStreamingRequest]) {
        if requests.is_empty() {
            return;
        }
        let mut backlog = self.streaming_backlog.write();
        backlog.extend(requests.iter().cloned());
        self.streaming_backlog_dirty
            .store(true, AtomicOrdering::Relaxed);
    }

    fn plan_priority(&self, kind: AssetStreamKind, id: usize) -> f32 {
        self.plan_entry(kind, id).unwrap_or(0.0)
    }

    fn effective_priority(&self, meta: &CacheEntryMeta) -> f32 {
        let current_epoch = self.current_plan_epoch();
        if meta.plan_epoch == current_epoch {
            meta.priority
        } else {
            0.0
        }
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
        let current_epoch = self.current_plan_epoch();
        let plan_priority = self.plan_entry(kind, id);
        let priority = plan_priority.unwrap_or(0.0);
        let plan_epoch = plan_priority.map(|_| current_epoch).unwrap_or(0);
        let (meta_map, _) = self.cache_meta_maps(kind);
        meta_map.write().insert(
            id,
            CacheEntryMeta {
                last_used: Instant::now(),
                size_bytes,
                priority,
                plan_epoch,
            },
        );
    }

    fn update_cache_size(&self, kind: AssetStreamKind, id: usize, size_bytes: usize) {
        let current_epoch = self.current_plan_epoch();
        let plan_priority = self.plan_entry(kind, id);
        let (meta_map, _) = self.cache_meta_maps(kind);
        let mut meta = meta_map.write();
        meta.entry(id)
            .and_modify(|m| {
                m.size_bytes = size_bytes;
                m.last_used = Instant::now();
                if let Some(priority) = plan_priority {
                    m.priority = priority;
                    m.plan_epoch = current_epoch;
                }
            })
            .or_insert(CacheEntryMeta {
                last_used: Instant::now(),
                size_bytes,
                priority: plan_priority.unwrap_or(0.0),
                plan_epoch: plan_priority.map(|_| current_epoch).unwrap_or(0),
            });
    }

    fn touch_cache_entry(&self, kind: AssetStreamKind, id: usize) {
        let current_epoch = self.current_plan_epoch();
        let plan_priority = self.plan_entry(kind, id);
        let (meta_map, _) = self.cache_meta_maps(kind);
        if let Some(meta) = meta_map.write().get_mut(&id) {
            meta.last_used = Instant::now();
            if let Some(priority) = plan_priority {
                meta.priority = priority;
                meta.plan_epoch = current_epoch;
            }
        }
    }

    fn record_audio_cache_entry(&self, id: usize, size_bytes: usize) {
        self.audio_meta.write().insert(
            id,
            CacheEntryMeta {
                last_used: Instant::now(),
                size_bytes,
                priority: 0.0,
                plan_epoch: 0,
            },
        );
    }

    fn touch_audio_cache_entry(&self, id: usize) {
        if let Some(meta) = self.audio_meta.write().get_mut(&id) {
            meta.last_used = Instant::now();
        }
    }

    fn enforce_audio_cache_budget(&self) {
        let budget = self.audio_budget;
        let mut meta = self.audio_meta.write();
        if budget == 0 {
            return;
        }
        let mut total: usize = meta.values().map(|m| m.size_bytes).sum();
        if total <= budget {
            return;
        }
        let mut entries: Vec<(Instant, usize, usize)> = meta
            .iter()
            .map(|(id, m)| (m.last_used, *id, m.size_bytes))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        for (_last, id, size) in entries {
            if total <= budget {
                break;
            }
            meta.remove(&id);
            self.audio_cache.write().remove(&id);
            total = total.saturating_sub(size);
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
            .map(|(id, m)| (self.effective_priority(m), m.last_used, *id, m.size_bytes))
            .collect();
        entries.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

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

    #[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(target_arch = "wasm32")]
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

        let mut released: Vec<usize> = Vec::new();
        let mut contexts = self.scene_contexts.write();

        if budget == 0 {
            for (scene_id, ctx) in contexts.iter_mut() {
                if ctx.pending_assets > 0 {
                    continue;
                }
                if active_scenes.contains(scene_id) {
                    continue;
                }
                if ctx.buffers_ready {
                    ctx.buffers_ready = false;
                    total = total.saturating_sub(ctx.buffers_bytes);
                    ctx.buffers_bytes = 0;
                    released.push(*scene_id);
                }
            }
            self.scene_buffer_bytes
                .store(total, AtomicOrdering::Relaxed);
        } else if total > budget {
            let mut entries: Vec<(Instant, usize, usize)> = contexts
                .iter()
                .filter_map(|(scene_id, ctx)| {
                    if ctx.pending_assets > 0 {
                        return None;
                    }
                    if active_scenes.contains(scene_id) {
                        return None;
                    }
                    if !ctx.buffers_ready {
                        return None;
                    }
                    Some((ctx.buffers_last_used, *scene_id, ctx.buffers_bytes))
                })
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

            for (_last_used, scene_id, bytes) in entries {
                if total <= budget {
                    break;
                }
                if let Some(ctx) = contexts.get_mut(&scene_id) {
                    if ctx.buffers_ready {
                        ctx.buffers_ready = false;
                        ctx.buffers_bytes = 0;
                        total = total.saturating_sub(bytes);
                        released.push(scene_id);
                    }
                }
            }
            self.scene_buffer_bytes
                .store(total, AtomicOrdering::Relaxed);
        }
        drop(contexts);
        for scene_id in released {
            helmer_worker_release_scene_buffers(scene_id as u32);
        }
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

    fn collect_idle_audio_ids(&self, idle: Duration, now: Instant, limit: usize) -> Vec<usize> {
        if limit == 0 {
            return Vec::new();
        }
        let mut ids = Vec::new();
        for (id, meta) in self.audio_meta.read().iter() {
            if ids.len() >= limit {
                break;
            }
            if now.duration_since(meta.last_used) < idle {
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
            remaining = remaining.saturating_sub(texture_ids.len());
        }

        if remaining == 0 {
            return;
        }

        let audio_ids = self.collect_idle_audio_ids(idle, now, remaining);
        if !audio_ids.is_empty() {
            let mut cache = self.audio_cache.write();
            let mut meta = self.audio_meta.write();
            for id in audio_ids.iter() {
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

    pub fn update_mesh(&self, id: usize, vertices: Vec<Vertex>, indices: Vec<u32>) {
        let request = AssetLoadRequest::ProceduralMesh {
            id,
            vertices,
            indices,
            tuning: self.asset_streaming_tuning,
        };
        if !self.enqueue_worker_request(request) {
            warn!(
                "Failed to queue procedural mesh update {}; worker thread offline",
                id
            );
        }
    }

    pub fn get_mesh(&self, id: usize) -> Option<Arc<Mesh>> {
        self.mesh_cache.read().get(&id).cloned()
    }

    pub fn get_audio(&self, id: usize) -> Option<Arc<AudioClip>> {
        let clip = self.audio_cache.read().get(&id).cloned();
        if clip.is_some() {
            self.touch_audio_cache_entry(id);
        }
        clip
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
        self.streaming_backlog_dirty
            .store(true, AtomicOrdering::Relaxed);
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

    pub fn set_asset_base_path(&self, path: impl Into<String>) {
        *self.asset_base_path.write() = Some(path.into());
    }

    pub fn clear_asset_base_path(&self) {
        *self.asset_base_path.write() = None;
    }

    pub fn asset_base_path(&self) -> Option<String> {
        self.asset_base_path.read().clone()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn set_opfs_enabled(&self, enabled: bool) {
        self.web_io.set_opfs_enabled(enabled);
        helmer_worker_set_opfs_enabled(enabled);
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

    pub fn audio_budget_bytes(&self) -> usize {
        self.audio_budget
    }

    pub fn set_audio_budget_bytes(&mut self, budget: usize) {
        self.audio_budget = budget;
        self.enforce_audio_cache_budget();
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

    pub fn audio_cache_usage_bytes(&self) -> usize {
        self.audio_meta.read().values().map(|m| m.size_bytes).sum()
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
                    let priority = mesh_meta
                        .get(id)
                        .map(|m| self.effective_priority(m))
                        .unwrap_or(0.0);
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
                    let priority = texture_meta
                        .get(id)
                        .map(|m| self.effective_priority(m))
                        .unwrap_or(0.0);
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
                    let priority = material_meta
                        .get(id)
                        .map(|m| self.effective_priority(m))
                        .unwrap_or(0.0);
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

    #[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(target_arch = "wasm32")]
    fn enqueue_worker_request(&self, request: AssetLoadRequest) -> bool {
        self.pending_worker_requests.write().push_back(request);
        true
    }

    #[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(target_arch = "wasm32")]
    fn flush_worker_requests(&self) {
        self.dispatch_web_requests();
    }

    #[cfg(target_arch = "wasm32")]
    fn dispatch_web_requests(&self) {
        self.web_worker_bridge
            .ensure_initialized(self.worker_queue_capacity);
        if !self.web_worker_bridge.is_ready() {
            return;
        }

        let max_inflight = {
            let base = self.worker_queue_capacity.max(1);
            let worker_count = self.web_worker_bridge.worker_count().max(1);
            let per_worker = WEB_MAX_INFLIGHT_PER_WORKER.max(1);
            base.min(worker_count.saturating_mul(per_worker).max(1))
        };
        let mut pending = self.pending_worker_requests.write();

        while self.inflight_worker_requests.load(AtomicOrdering::Relaxed) < max_inflight {
            let Some(request) = pending.pop_front() else {
                break;
            };
            let Some(dispatch) = Self::worker_request_from_asset(&request) else {
                continue;
            };
            let payload = match bincode::serialize(&dispatch.request) {
                Ok(payload) => payload,
                Err(err) => {
                    warn!("Failed to serialize worker request: {}", err);
                    continue;
                }
            };

            let sent = if let Some(scene_id) = dispatch.scene_affinity {
                let worker_index = self.worker_index_for_scene(scene_id);
                helmer_worker_enqueue_on_worker(worker_index as u32, &payload)
            } else {
                helmer_worker_enqueue(&payload)
            };

            if !sent {
                warn!("Failed to enqueue worker request; deferring.");
                pending.push_front(request);
                self.web_worker_bridge.mark_failed();
                break;
            }
            self.inflight_worker_requests
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn worker_request_from_asset(request: &AssetLoadRequest) -> Option<WebWorkerDispatch> {
        match request {
            AssetLoadRequest::Mesh { id, path, tuning } => Some(WorkerRequest::Mesh {
                id: *id,
                path: path.to_string_lossy().to_string(),
                tuning: *tuning,
            })
            .map(|request| WebWorkerDispatch {
                request,
                scene_affinity: None,
            }),
            AssetLoadRequest::ProceduralMesh {
                id,
                vertices,
                indices,
                tuning,
            } => Some(WorkerRequest::ProceduralMesh {
                id: *id,
                vertices: cast_slice(vertices.as_slice()).to_vec(),
                indices: indices.clone(),
                tuning: *tuning,
            })
            .map(|request| WebWorkerDispatch {
                request,
                scene_affinity: None,
            }),
            AssetLoadRequest::Texture { id, path, kind } => Some(WorkerRequest::Texture {
                id: *id,
                path: path.to_string_lossy().to_string(),
                kind: *kind,
            })
            .map(|request| WebWorkerDispatch {
                request,
                scene_affinity: None,
            }),
            AssetLoadRequest::Material { id, path } => Some(WorkerRequest::Material {
                id: *id,
                path: path.to_string_lossy().to_string(),
            })
            .map(|request| WebWorkerDispatch {
                request,
                scene_affinity: None,
            }),
            AssetLoadRequest::Audio { .. } => None,
            AssetLoadRequest::Scene { id, path } => Some(WorkerRequest::Scene {
                id: *id,
                path: path.to_string_lossy().to_string(),
            })
            .map(|request| WebWorkerDispatch {
                request,
                scene_affinity: Some(*id),
            }),
            AssetLoadRequest::SceneBuffers { scene_id, .. } => Some(WorkerRequest::SceneBuffers {
                scene_id: *scene_id,
            })
            .map(|request| WebWorkerDispatch {
                request,
                scene_affinity: Some(*scene_id),
            }),
            AssetLoadRequest::StreamMesh {
                id,
                scene_id,
                desc,
                tuning,
                ..
            } => Some(WorkerRequest::StreamMesh {
                id: *id,
                scene_id: *scene_id,
                desc: *desc,
                tuning: *tuning,
            })
            .map(|request| WebWorkerDispatch {
                request,
                scene_affinity: Some(*scene_id),
            }),
            AssetLoadRequest::StreamTexture {
                id,
                scene_id,
                tex_index,
                kind,
                ..
            } => Some(WorkerRequest::StreamTexture {
                id: *id,
                scene_id: *scene_id,
                tex_index: *tex_index,
                kind: *kind,
            })
            .map(|request| WebWorkerDispatch {
                request,
                scene_affinity: Some(*scene_id),
            }),
            AssetLoadRequest::LowResTexture {
                id,
                name,
                kind,
                data,
                format,
                dimensions,
                max_dim,
            } => {
                let Some(worker_format) = WorkerTextureFormat::from_wgpu(*format) else {
                    warn!(
                        "Unsupported low-res texture format for request {} ({:?})",
                        id, format
                    );
                    return None;
                };
                Some(WorkerRequest::LowResTexture {
                    id: *id,
                    name: name.clone(),
                    kind: *kind,
                    data: data.as_ref().to_vec(),
                    format: worker_format,
                    dimensions: *dimensions,
                    max_dim: *max_dim,
                })
                .map(|request| WebWorkerDispatch {
                    request,
                    scene_affinity: None,
                })
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn worker_index_for_scene(&self, scene_id: usize) -> usize {
        let count = self.web_worker_bridge.worker_count().max(1);
        scene_id % count
    }

    fn is_remote_path(path: &str) -> bool {
        path.starts_with("http://")
            || path.starts_with("https://")
            || path.starts_with("blob:")
            || path.starts_with("data:")
    }

    fn is_url_base(base: &str) -> bool {
        base.contains("://") || base.starts_with("blob:") || base.starts_with("data:")
    }

    fn resolve_asset_path(&self, path: &Path) -> PathBuf {
        let path_str = path.to_string_lossy();
        if path.is_absolute() || Self::is_remote_path(&path_str) {
            return path.to_path_buf();
        }

        let Some(base) = self.asset_base_path.read().clone() else {
            return path.to_path_buf();
        };
        if base.is_empty() {
            return path.to_path_buf();
        }

        if Self::is_url_base(&base) {
            let trimmed_base = base.trim_end_matches('/');
            let trimmed_path = path_str.trim_start_matches("./").trim_start_matches('/');
            let joined = if trimmed_base.is_empty() {
                trimmed_path.to_string()
            } else if trimmed_path.is_empty() {
                trimmed_base.to_string()
            } else {
                format!("{}/{}", trimmed_base, trimmed_path)
            };
            return PathBuf::from(joined);
        }

        PathBuf::from(base).join(path)
    }

    pub fn load_mesh<P: AsRef<Path>>(&self, path: P) -> Handle<Mesh> {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        let resolved_path = self.resolve_asset_path(path.as_ref());
        let request = AssetLoadRequest::Mesh {
            id,
            path: resolved_path,
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
        let resolved_path = self.resolve_asset_path(path.as_ref());
        let request = AssetLoadRequest::Texture {
            id,
            path: resolved_path,
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
        let resolved_path = self.resolve_asset_path(path.as_ref());
        let request = AssetLoadRequest::Material {
            id,
            path: resolved_path,
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

    pub fn load_audio<P: AsRef<Path>>(&self, path: P, mode: AudioLoadMode) -> Handle<AudioClip> {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        let resolved_path = self.resolve_asset_path(path.as_ref());
        let request = AssetLoadRequest::Audio {
            id,
            path: resolved_path.clone(),
            mode,
        };
        if !self.enqueue_worker_request(request) {
            warn!(
                "Failed to queue audio request {}; worker thread offline",
                id
            );
        }
        self.audio_sources.write().insert(
            id,
            AudioSource {
                path: resolved_path,
                mode,
            },
        );
        Handle::new(id)
    }

    pub fn load_audio_static<P: AsRef<Path>>(&self, path: P) -> Handle<AudioClip> {
        self.load_audio(path, AudioLoadMode::Static)
    }

    pub fn load_audio_streaming<P: AsRef<Path>>(&self, path: P) -> Handle<AudioClip> {
        self.load_audio(path, AudioLoadMode::Streaming)
    }

    pub fn load_scene<P: AsRef<Path>>(&self, path: P) -> Handle<Scene> {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        let resolved_path = self.resolve_asset_path(path.as_ref());
        let request = AssetLoadRequest::Scene {
            id,
            path: resolved_path,
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

    #[cfg(target_arch = "wasm32")]
    pub fn store_virtual_asset<P: AsRef<Path>>(&self, path: P, bytes: Vec<u8>) {
        let resolved_path = self.resolve_asset_path(path.as_ref());
        let path_str = resolved_path.to_string_lossy();
        helmer_worker_store_virtual_asset(path_str.as_ref(), &bytes);
        self.web_io.store_virtual_asset_path(&resolved_path, bytes);
    }

    pub fn get_scene(&self, handle: &Handle<Scene>) -> Option<Arc<Scene>> {
        self.scenes.read().get(&handle.id).cloned()
    }

    fn handle_asset_result(&self, result: AssetLoadResult) {
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
                    .and_then(|_| self.take_streaming_inflight(AssetStreamKind::Mesh, id))
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
                    return;
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
                    .and_then(|_| self.take_streaming_inflight(AssetStreamKind::Texture, id))
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
                            match self.try_send_asset_message(RenderMessage::CreateTexture {
                                id,
                                name: tex.name.clone(),
                                kind: tex.kind,
                                data: low.data.clone(),
                                format: low.format,
                                width: low.dimensions.0,
                                height: low.dimensions.1,
                            }) {
                                Ok(()) => {
                                    self.touch_cache_entry(AssetStreamKind::Texture, id);
                                }
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
                            warn!("Failed to send texture {}; render thread offline", id);
                        }
                    }
                }
            }
            AssetLoadResult::Material { id, data } => {
                let material_gpu_data = self.resolve_material_dependencies(id, data);
                let cached = self.material_budget > 0;
                if cached {
                    let size_bytes = std::mem::size_of_val(&material_gpu_data);
                    self.material_cache
                        .write()
                        .insert(id, Arc::new(material_gpu_data.clone()));
                    self.record_cache_entry(AssetStreamKind::Material, id, size_bytes);
                    self.enforce_cache_budget(AssetStreamKind::Material);
                }
                let request = AssetStreamingRequest {
                    id,
                    kind: AssetStreamKind::Material,
                    priority: 0.0,
                    max_lod: None,
                    force_low_res: false,
                };
                match self.try_send_asset_message(RenderMessage::CreateMaterial(material_gpu_data))
                {
                    Ok(()) => {
                        if cached {
                            self.touch_cache_entry(AssetStreamKind::Material, id);
                        }
                    }
                    Err(AssetSendError::Budget) | Err(AssetSendError::Full) => {
                        self.requeue_stream_request(request);
                    }
                    Err(AssetSendError::Disconnected) => {
                        warn!("Failed to send material {}; render thread offline", id);
                    }
                }
            }
            AssetLoadResult::Audio { id, clip } => {
                let size_bytes = clip.size_bytes;
                self.audio_cache.write().insert(id, Arc::new(clip));
                self.record_audio_cache_entry(id, size_bytes);
                self.enforce_audio_cache_budget();
            }
            AssetLoadResult::Scene { id: scene_id, data } => {
                self.register_scene(scene_id, data);
            }
            AssetLoadResult::SceneBuffers { scene_id, payload } => {
                self.scene_buffer_inflight.write().remove(&scene_id);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let SceneBuffersPayload {
                        buffers,
                        buffers_bytes,
                    } = payload;
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
                    self.maybe_populate_scene_skinning(scene_id);
                }
                #[cfg(target_arch = "wasm32")]
                {
                    let buffers_bytes = payload.buffers_bytes;
                    let stored = {
                        let mut contexts = self.scene_contexts.write();
                        let mut stored = false;
                        if let Some(ctx) = contexts.get_mut(&scene_id) {
                            if !ctx.buffers_ready {
                                ctx.buffers_ready = true;
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

    pub fn update(&self) {
        self.flush_worker_requests();
        #[cfg(target_arch = "wasm32")]
        let update_start = Instant::now();
        #[cfg(target_arch = "wasm32")]
        let update_budget = Duration::from_millis(2);
        // --- 1. Drain all completed load results from workers ---
        let mut creation_budget = self.asset_creation_limit_per_frame;
        // Limit how many worker results we process per tick to avoid long stalls on huge loads
        let mut results_budget = {
            #[cfg(target_arch = "wasm32")]
            {
                self.asset_creation_limit_per_frame.max(1)
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.asset_creation_limit_per_frame.saturating_mul(4).max(4)
            }
        };
        while results_budget > 0 {
            #[cfg(target_arch = "wasm32")]
            {
                if update_start.elapsed() >= update_budget {
                    break;
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                let pending_bytes = {
                    let mut pending = self.web_pending_responses.lock();
                    pending.pop_front()
                };
                if let Some(bytes) = pending_bytes {
                    results_budget = results_budget.saturating_sub(1);
                    match bincode::deserialize::<WorkerResponse>(&bytes) {
                        Ok(response) => {
                            if let Some(result) = worker_response_to_result(response) {
                                self.handle_asset_result(result);
                            }
                        }
                        Err(err) => {
                            warn!("Failed to decode worker response: {}", err);
                        }
                    }
                    let _ = self.inflight_worker_requests.fetch_update(
                        AtomicOrdering::Relaxed,
                        AtomicOrdering::Relaxed,
                        |value| Some(value.saturating_sub(1)),
                    );
                    continue;
                }
            }
            match self.result_receiver.try_recv() {
                Ok(result) => {
                    results_budget = results_budget.saturating_sub(1);
                    self.handle_asset_result(result);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            if update_start.elapsed() >= update_budget {
                return;
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

    #[cfg(not(target_arch = "wasm32"))]
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
            skins_loaded: false,
            animations_loaded: false,
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
                    alpha_mode: mat.alpha_mode,
                    alpha_cutoff: mat.alpha_cutoff,
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
                    skin_index: node.skin_index,
                    node_index: node.node_index,
                })
            })
            .collect();

        let skins = Arc::new(RwLock::new(Vec::new()));
        let animations = Arc::new(RwLock::new(Vec::new()));
        self.scenes.write().insert(
            scene_id,
            Arc::new(Scene {
                nodes: scene_nodes,
                skins: skins.clone(),
                animations: animations.clone(),
            }),
        );

        self.maybe_populate_scene_skinning(scene_id);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn maybe_populate_scene_skinning(&self, scene_id: usize) {
        let (doc, buffers, should_parse) = {
            let contexts = self.scene_contexts.read();
            let Some(ctx) = contexts.get(&scene_id) else {
                return;
            };
            let buffers = ctx.buffers.clone();
            let should_parse = buffers.is_some() && !(ctx.skins_loaded && ctx.animations_loaded);
            (ctx.doc.clone(), buffers, should_parse)
        };
        if !should_parse {
            return;
        }
        let Some(buffers) = buffers else {
            return;
        };

        let parent_map = build_parent_map(&doc);
        let skin_count = doc.skins().len();
        let mut skins: Vec<Arc<Skin>> = Vec::with_capacity(skin_count);
        let mut joint_maps: Vec<HashMap<usize, usize>> = Vec::with_capacity(skin_count);

        for skin in doc.skins() {
            let name = skin
                .name()
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("skin_{}", skin.index()));
            let joints: Vec<usize> = skin.joints().map(|node| node.index()).collect();
            let joint_map: HashMap<usize, usize> = joints
                .iter()
                .enumerate()
                .map(|(idx, node)| (*node, idx))
                .collect();
            let reader = skin.reader(|buffer| buffers.get(buffer.index()).map(|b| b.as_slice()));
            let inverse_bind_matrices: Vec<Mat4> = reader
                .read_inverse_bind_matrices()
                .map(|iter| iter.map(|mat| Mat4::from_cols_array_2d(&mat)).collect())
                .unwrap_or_default();

            let mut skeleton_joints = Vec::with_capacity(joints.len());
            for (joint_index, node_index) in joints.iter().enumerate() {
                let node = doc.nodes().nth(*node_index);
                let bind_transform = node
                    .as_ref()
                    .map(|node| {
                        Transform::from_matrix(Mat4::from_cols_array_2d(&node.transform().matrix()))
                    })
                    .unwrap_or_default();
                let parent = parent_map
                    .get(*node_index)
                    .and_then(|p| *p)
                    .and_then(|parent_node| joint_map.get(&parent_node).copied());
                let inverse_bind = inverse_bind_matrices
                    .get(joint_index)
                    .copied()
                    .unwrap_or(Mat4::IDENTITY);
                let joint_name = node
                    .and_then(|node| node.name().map(|s| s.to_string()))
                    .unwrap_or_else(|| format!("joint_{}", node_index));
                skeleton_joints.push(Joint {
                    name: joint_name,
                    parent,
                    bind_transform,
                    inverse_bind,
                });
            }

            let skeleton = Skeleton::new(skeleton_joints);
            let joint_names = skeleton
                .joints
                .iter()
                .map(|joint| joint.name.clone())
                .collect();
            skins.push(Arc::new(Skin {
                name,
                skeleton: Arc::new(skeleton),
                joint_names,
            }));
            joint_maps.push(joint_map);
        }

        let mut animations: Vec<Arc<AnimationLibrary>> = Vec::with_capacity(skin_count);
        for _ in 0..skin_count {
            animations.push(Arc::new(AnimationLibrary::default()));
        }
        if skin_count > 0 && doc.animations().len() > 0 {
            let mut libraries: Vec<AnimationLibrary> = (0..skin_count)
                .map(|_| AnimationLibrary::default())
                .collect();
            for animation in doc.animations() {
                let anim_name = animation
                    .name()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("anim_{}", animation.index()));
                let parsed_channels = parse_animation_channels(&animation, buffers.as_ref());
                if parsed_channels.is_empty() {
                    continue;
                }
                let mut durations = vec![0.0f32; skin_count];
                for channel in parsed_channels.iter() {
                    for (skin_index, map) in joint_maps.iter().enumerate() {
                        if let Some(&joint_index) = map.get(&channel.node_index) {
                            let mut channel = channel.channel.clone();
                            retarget_channel(&mut channel, joint_index);
                            let clip = libraries[skin_index].name_to_index.get(&anim_name).copied();
                            let clip_index = clip.unwrap_or_else(|| {
                                libraries[skin_index].add_clip(AnimationClip {
                                    name: anim_name.clone(),
                                    duration: 0.0,
                                    channels: Vec::new(),
                                })
                            });
                            if let Some(existing) =
                                libraries[skin_index].clips.get(clip_index).cloned()
                            {
                                let mut clip = (*existing).clone();
                                clip.channels.push(channel);
                                clip.duration = clip
                                    .duration
                                    .max(channel_duration(&clip.channels.last().unwrap()));
                                let clip_duration = clip.duration;
                                libraries[skin_index].clips[clip_index] = Arc::new(clip);
                                durations[skin_index] = durations[skin_index].max(clip_duration);
                            }
                        }
                    }
                }
                for (idx, duration) in durations.into_iter().enumerate() {
                    if duration <= 0.0 {
                        continue;
                    }
                    if let Some(clip_index) = libraries[idx].clip_index(&anim_name) {
                        let clip = libraries[idx].clips[clip_index].as_ref().clone();
                        let mut clip = clip.clone();
                        clip.duration = duration;
                        libraries[idx].clips[clip_index] = Arc::new(clip);
                    }
                }
            }
            animations = libraries.into_iter().map(Arc::new).collect();
        }

        if let Some(scene) = self.scenes.read().get(&scene_id).cloned() {
            if skin_count > 0 {
                *scene.skins.write() = skins;
                *scene.animations.write() = animations;
            }
        }

        let mut contexts = self.scene_contexts.write();
        if let Some(ctx) = contexts.get_mut(&scene_id) {
            ctx.skins_loaded = true;
            ctx.animations_loaded = true;
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn maybe_populate_scene_skinning(&self, _scene_id: usize) {}

    #[cfg(target_arch = "wasm32")]
    fn register_scene(
        &self,
        scene_id: usize,
        data: crate::runtime::asset_worker::WorkerSceneSummary,
    ) {
        info!(
            "Scene {} parsed; registering layout for streaming-based loading.",
            scene_id
        );

        let crate::runtime::asset_worker::WorkerSceneSummary {
            buffers_bytes,
            buffers_ready,
            base_path,
            scene_path,
            textures,
            materials,
            mesh_primitives,
            nodes,
            mesh_bounds,
        } = data;

        let context = SceneAssetContext {
            buffers_ready,
            buffers_bytes,
            buffers_last_used: Instant::now(),
            base_path: base_path.map(PathBuf::from),
            scene_path: scene_path.map(PathBuf::from),
            pending_assets: textures.len() + materials.len() + mesh_primitives.len(),
            skins_loaded: false,
            animations_loaded: false,
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
                    alpha_mode: mat.alpha_mode,
                    alpha_cutoff: mat.alpha_cutoff,
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

        if !mesh_bounds.is_empty() {
            let mut map = self.mesh_aabb_map.write();
            for (idx, mesh_id) in mesh_ids.iter().enumerate() {
                let bounds = mesh_bounds
                    .get(idx)
                    .map(|bounds| Aabb {
                        min: Vec3::from(bounds.min),
                        max: Vec3::from(bounds.max),
                    })
                    .unwrap_or(Aabb {
                        min: Vec3::ZERO,
                        max: Vec3::ZERO,
                    });
                map.0.insert(*mesh_id, bounds);
            }
        }

        let scene_nodes = nodes
            .iter()
            .enumerate()
            .filter_map(|(node_index, node)| {
                let mesh_id = mesh_ids.get(node.primitive_desc_index)?;
                let material_id = material_ids.get(node.material_index)?;
                Some(SceneNode {
                    mesh: Handle::new(*mesh_id),
                    material: Handle::new(*material_id),
                    transform: Mat4::from_cols_array(&node.transform),
                    skin_index: None,
                    node_index,
                })
            })
            .collect();

        let skins = Arc::new(RwLock::new(Vec::new()));
        let animations = Arc::new(RwLock::new(Vec::new()));
        self.scenes.write().insert(
            scene_id,
            Arc::new(Scene {
                nodes: scene_nodes,
                skins,
                animations,
            }),
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn release_scene_buffers_if_needed(&self, buffers: &[StreamedBuffer]) {
        if self.scene_buffer_budget != 0 {
            return;
        }
        for buffer in buffers {
            buffer.release_pages();
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn release_scene_buffers_if_needed(&self, _buffers: &[StreamedBuffer]) {}

    #[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(target_arch = "wasm32")]
    fn drop_scene_buffers(&self, scene_id: usize) {
        let mut contexts = self.scene_contexts.write();
        let Some(ctx) = contexts.get_mut(&scene_id) else {
            return;
        };
        if !ctx.buffers_ready {
            return;
        }
        ctx.buffers_ready = false;
        let bytes = ctx.buffers_bytes;
        ctx.buffers_bytes = 0;
        if bytes > 0 {
            let total = self.scene_buffer_bytes.load(AtomicOrdering::Relaxed);
            self.scene_buffer_bytes
                .store(total.saturating_sub(bytes), AtomicOrdering::Relaxed);
        }
        helmer_worker_release_scene_buffers(scene_id as u32);
    }

    #[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(target_arch = "wasm32")]
    fn scene_context_status(&self, scene_id: usize) -> SceneContextStatus {
        let ready = {
            let mut contexts = self.scene_contexts.write();
            let Some(ctx) = contexts.get_mut(&scene_id) else {
                return SceneContextStatus::Missing;
            };
            if ctx.buffers_ready {
                ctx.buffers_last_used = Instant::now();
                true
            } else {
                false
            }
        };

        if ready {
            SceneContextStatus::Ready
        } else {
            let _ = self.queue_scene_buffer_load(scene_id);
            SceneContextStatus::Pending
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(target_arch = "wasm32")]
    fn queue_scene_buffer_load(&self, scene_id: usize) -> bool {
        {
            let inflight = self.scene_buffer_inflight.read();
            if inflight.contains(&scene_id) {
                return true;
            }
        }

        let exists = {
            let contexts = self.scene_contexts.read();
            contexts.contains_key(&scene_id)
        };
        if !exists {
            return false;
        }

        let request = AssetLoadRequest::SceneBuffers { scene_id };
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

    #[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(target_arch = "wasm32")]
    fn mark_scene_asset_complete(&self, scene_id: usize) {
        let budget = self.scene_buffer_budget;
        let keep_buffers = budget == 0 && self.scene_has_streaming_activity(scene_id);
        let mut release = false;
        {
            let mut contexts = self.scene_contexts.write();
            let Some(ctx) = contexts.get_mut(&scene_id) else {
                return;
            };
            if ctx.pending_assets > 0 {
                ctx.pending_assets -= 1;
            }
            if budget == 0 && ctx.pending_assets == 0 && ctx.buffers_ready {
                if keep_buffers {
                    return;
                }
                ctx.buffers_ready = false;
                let bytes = ctx.buffers_bytes;
                ctx.buffers_bytes = 0;
                if bytes > 0 {
                    let total = self.scene_buffer_bytes.load(AtomicOrdering::Relaxed);
                    self.scene_buffer_bytes
                        .store(total.saturating_sub(bytes), AtomicOrdering::Relaxed);
                }
                release = true;
            }
        }
        if release {
            helmer_worker_release_scene_buffers(scene_id as u32);
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
        let existing_priority = existing.priority;
        existing.priority = existing.priority.max(incoming.priority);
        existing.max_lod = match (existing.max_lod, incoming.max_lod) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (None, _) | (_, None) => None,
        };
        if incoming.priority > existing_priority {
            existing.force_low_res = incoming.force_low_res;
        } else if incoming.priority == existing_priority {
            existing.force_low_res &= incoming.force_low_res;
        }
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
        if self.streaming_backlog_limit == 0 {
            return;
        }
        self.streaming_backlog_dirty
            .store(true, AtomicOrdering::Relaxed);
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

        let plan_epoch = self.streaming_plan_epoch.load(AtomicOrdering::Relaxed);
        let backlog_epoch = self.streaming_backlog_epoch.load(AtomicOrdering::Relaxed);
        let backlog_dirty = self
            .streaming_backlog_dirty
            .swap(false, AtomicOrdering::Relaxed);
        let limit = self.streaming_backlog_limit;
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

        if limit == 0 {
            backlog.clear();
            self.streaming_backlog_epoch
                .store(plan_epoch, AtomicOrdering::Relaxed);
            return;
        }

        let plan_changed = plan_epoch != backlog_epoch;
        if !had_new && !backlog_dirty && !plan_changed {
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
                .total_cmp(&a.priority)
                .then_with(|| a.id.cmp(&b.id))
                .then_with(|| Self::stream_kind_rank(a.kind).cmp(&Self::stream_kind_rank(b.kind)))
        });
        if merged_vec.len() > limit {
            merged_vec.truncate(limit);
        }
        *backlog = VecDeque::from(merged_vec);
        self.streaming_backlog_epoch
            .store(plan_epoch, AtomicOrdering::Relaxed);
    }

    fn process_stream_requests(&self, limit: usize, creation_budget: &mut usize) {
        self.enqueue_stream_requests();

        let mut backlog = self.streaming_backlog.write();
        let mut sent = 0usize;
        let mut attempts = 0usize;
        let total = backlog.len();
        #[cfg(target_arch = "wasm32")]
        let max_attempts = total.min(
            limit
                .max(1)
                .saturating_mul(WEB_STREAMING_ATTEMPT_MULTIPLIER.max(1)),
        );
        #[cfg(not(target_arch = "wasm32"))]
        let max_attempts = total;

        while sent < limit && attempts < max_attempts {
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

                #[cfg(not(target_arch = "wasm32"))]
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
                #[cfg(target_arch = "wasm32")]
                {
                    match self.scene_context_status(pending.scene_id) {
                        SceneContextStatus::Ready => {}
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
                    }
                }
                self.update_streaming_inflight(request);
                let queued = self.enqueue_worker_request(AssetLoadRequest::StreamMesh {
                    id: request.id,
                    scene_id: pending.scene_id,
                    desc: pending.desc,
                    #[cfg(not(target_arch = "wasm32"))]
                    doc,
                    #[cfg(not(target_arch = "wasm32"))]
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

                #[cfg(not(target_arch = "wasm32"))]
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
                #[cfg(target_arch = "wasm32")]
                {
                    match self.scene_context_status(pending.scene_id) {
                        SceneContextStatus::Ready => {}
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
                    }
                }
                self.update_streaming_inflight(request);
                let queued = self.enqueue_worker_request(AssetLoadRequest::StreamTexture {
                    id: request.id,
                    scene_id: pending.scene_id,
                    tex_index: pending.request.tex_index,
                    kind: pending.request.kind,
                    #[cfg(not(target_arch = "wasm32"))]
                    doc,
                    #[cfg(not(target_arch = "wasm32"))]
                    buffers,
                    #[cfg(not(target_arch = "wasm32"))]
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
            alpha_mode: data.alpha_mode,
            alpha_cutoff: data.alpha_cutoff,
        }
    }
}

// --- WORKER HELPER FUNCTIONS ---

fn request_path(req: &AssetLoadRequest) -> &Path {
    match req {
        AssetLoadRequest::Mesh { path, .. } => path,
        AssetLoadRequest::Texture { path, .. } => path,
        AssetLoadRequest::Material { path, .. } => path,
        AssetLoadRequest::Audio { path, .. } => path,
        AssetLoadRequest::Scene { path, .. } => path,
        #[cfg(not(target_arch = "wasm32"))]
        AssetLoadRequest::SceneBuffers { scene_path, .. } => {
            scene_path.as_deref().unwrap_or_else(|| Path::new(""))
        }
        _ => Path::new(""),
    }
}

#[cfg(not(target_arch = "wasm32"))]
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

pub(crate) fn is_gltf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("gltf"))
        .unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
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
                    if uri_is_remote(uri) {
                        return Err(format!(
                            "Remote buffer URI '{}' is not supported on native target",
                            uri
                        ));
                    }
                    let buffer_uri = normalize_local_uri_reference(uri);
                    let buf_path = base.join(&buffer_uri);
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

#[cfg(not(target_arch = "wasm32"))]
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
                    if uri_is_remote(uri) {
                        return Err(format!(
                            "Remote buffer URI '{}' is not supported on native target",
                            uri
                        ));
                    }
                    let buffer_uri = normalize_local_uri_reference(uri);
                    let buf_path = base.join(&buffer_uri);
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

#[cfg(target_arch = "wasm32")]
async fn load_and_parse_web<T, F, R>(
    id: usize,
    path: &Path,
    io: &WebAssetIo,
    parse_fn: F,
    result_fn: R,
) -> Option<AssetLoadResult>
where
    F: Fn(&[u8]) -> Result<T, String>,
    R: Fn(usize, T) -> AssetLoadResult,
{
    match io.read_path(path).await {
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

#[cfg(target_arch = "wasm32")]
pub(crate) async fn load_gltf_streaming_web(
    path: &Path,
    io: &WebAssetIo,
) -> Result<(gltf::Document, Vec<StreamedBuffer>), String> {
    let bytes = io.read_path(path).await?;
    let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| e.to_string())?;
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
                    let resolved_uri = normalize_uri_for_external_read(uri, base_dir);
                    let bytes = io.read_uri(base_dir, &resolved_uri).await?;
                    StreamedBuffer::owned(bytes)
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

#[cfg(target_arch = "wasm32")]
pub(crate) async fn load_scene_buffers_web(
    doc: &gltf::Document,
    scene_path: Option<&Path>,
    base_path: Option<&Path>,
    io: &WebAssetIo,
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
                    let bytes = io.read_path(path).await?;
                    let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| e.to_string())?;
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
                    let resolved_uri = normalize_uri_for_external_read(uri, base_dir);
                    let bytes = io.read_uri(base_dir, &resolved_uri).await?;
                    StreamedBuffer::owned(bytes)
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

#[cfg(target_arch = "wasm32")]
async fn read_image_bytes_from_source_web<'a, B: BufferSource>(
    source: &'a gltf::image::Source<'a>,
    buffers: &'a [B],
    base_path: Option<&'a Path>,
    io: &WebAssetIo,
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
                let ext_hint = extension_hint_from_uri(uri);
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
            let resolved_uri = normalize_uri_for_external_read(uri, Some(base));
            match io.read_uri(Some(base), &resolved_uri).await {
                Ok(bytes) => {
                    let ext_hint = extension_hint_from_uri(uri)
                        .or_else(|| extension_hint_from_path(Path::new(&resolved_uri)));
                    Ok(Some((Cow::Owned(bytes), mime_hint, ext_hint)))
                }
                Err(e) => Err(format!(
                    "Failed to read external image '{}' (base {:?}): {}",
                    uri, base, e
                )),
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn decode_texture_asset_web(
    gltf_texture: gltf::Texture<'_>,
    buffers: &[StreamedBuffer],
    base_path: Option<&Path>,
    io: &WebAssetIo,
    kind: AssetKind,
) -> Option<(String, AssetKind, Vec<u8>, wgpu::TextureFormat, (u32, u32))> {
    let image = gltf_texture.source();
    let name = gltf_texture
        .name()
        .unwrap_or("unnamed_gltf_texture")
        .to_string();

    let image_source = image.source();
    let (pixels, mime_hint, ext_hint) =
        match read_image_bytes_from_source_web(&image_source, buffers, base_path, io).await {
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

#[cfg(not(target_arch = "wasm32"))]
fn parse_mesh_from_gltf_path(path: &Path) -> Result<(Vec<Vertex>, Vec<u32>, Aabb), String> {
    let (document, buffers) = load_gltf_streaming(path)?;
    parse_mesh_document(&document, &buffers)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_scene_from_gltf_path(path: &Path) -> Result<ParsedGltfScene, String> {
    let gltf = gltf::Gltf::open(path).map_err(|e| e.to_string())?;
    parse_scene_document(gltf.document, None, path.parent(), Some(path.to_path_buf()))
}

#[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(not(target_arch = "wasm32"))]
    let thread_count = std::thread::available_parallelism()
        .map(|count| count.get() as u32)
        .unwrap_or(1)
        .min(4)
        .max(1);
    #[cfg(target_arch = "wasm32")]
    let thread_count = 1;
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

pub(crate) fn decode_ktx2(
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

    let (target_transcode_format, target_wgpu_format) = if cfg!(target_arch = "wasm32") {
        let format = match kind {
            AssetKind::Albedo | AssetKind::Emission => wgpu::TextureFormat::Rgba8UnormSrgb,
            _ => wgpu::TextureFormat::Rgba8Unorm,
        };
        (TranscoderTextureFormat::RGBA32, format)
    } else {
        let format = match kind {
            AssetKind::Albedo | AssetKind::Emission => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
            _ => wgpu::TextureFormat::Bc7RgbaUnorm,
        };
        (TranscoderTextureFormat::BC7_RGBA, format)
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

    #[cfg(target_arch = "wasm32")]
    {
        let level_data = transcoder
            .transcode_image_level(bytes, target_transcode_format, transcode_params)
            .map_err(|e| format!("Failed to transcode KTX2 base level for web: {:?}", e))?;
        transcoder.end_transcoding();
        return Ok((level_data, target_wgpu_format, dimensions));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
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
                .map_err(|e| {
                    format!("Failed to transcode generated mip level {}: {:?}", level, e)
                })?;
            combined.extend_from_slice(&level_data);
        }
        mip_transcoder.end_transcoding();

        Ok((combined, target_wgpu_format, dimensions))
    }
}

pub(crate) fn parse_ron_material(bytes: &[u8]) -> Result<MaterialFile, String> {
    ron::de::from_bytes(bytes).map_err(|e| e.to_string())
}

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

#[cfg(target_arch = "wasm32")]
static MESHOPT_RUNTIME_OK: AtomicBool = AtomicBool::new(true);
#[cfg(target_arch = "wasm32")]
static MESHOPT_VCACHE_WARNED: AtomicBool = AtomicBool::new(false);

#[cfg(target_arch = "wasm32")]
fn meshopt_runtime_ok() -> bool {
    MESHOPT_RUNTIME_OK.load(AtomicOrdering::Relaxed)
}

#[cfg(target_arch = "wasm32")]
fn disable_meshopt_runtime(reason: &str) {
    if MESHOPT_RUNTIME_OK.swap(false, AtomicOrdering::Relaxed) {
        warn!("meshopt disabled on wasm: {}", reason);
    }
}

fn meshopt_enabled() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        web_global_bool("HELMER_WASM_MESHOPT").unwrap_or(false) && meshopt_runtime_ok()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("HELMER_ENABLE_MESHOPT")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true)
    }
}

fn meshopt_vertex_cache_enabled() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let requested = web_global_bool("HELMER_WASM_MESHOPT_VCACHE").unwrap_or(false);
        if requested && !MESHOPT_VCACHE_WARNED.swap(true, AtomicOrdering::Relaxed) {
            warn!("HELMER_WASM_MESHOPT_VCACHE is disabled on wasm (known to break meshes).");
        }
        false
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        true
    }
}

fn meshopt_simplify_enabled() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        web_global_bool("HELMER_WASM_MESHOPT_SIMPLIFY").unwrap_or(false)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        true
    }
}

fn meshopt_budget_bytes() -> usize {
    const DEFAULT_BUDGET_MB: usize = 128;
    let budget_mb = std::env::var("HELMER_MESHOPT_BUDGET_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_BUDGET_MB);
    budget_mb.saturating_mul(1024 * 1024)
}

#[derive(Clone, Copy, Debug)]
enum MeshoptCacheAlgo {
    Forsyth,
    Fifo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MeshoptSimplifyKind {
    Standard,
    Sloppy,
}

fn meshopt_cache_algo() -> MeshoptCacheAlgo {
    #[cfg(target_arch = "wasm32")]
    {
        // Web builds stay on FIFO to avoid table-path instability and keep behavior consistent.
        return MeshoptCacheAlgo::Fifo;
    }
    #[cfg(not(target_arch = "wasm32"))]
    std::env::var("HELMER_MESHOPT_CACHE_ALGO")
        .ok()
        .map(|v| v.to_lowercase())
        .as_deref()
        .map(|v| match v {
            "forsyth" => MeshoptCacheAlgo::Forsyth,
            _ => MeshoptCacheAlgo::Fifo,
        })
        .unwrap_or(MeshoptCacheAlgo::Fifo)
}

fn meshopt_cache_size() -> u32 {
    const DEFAULT_CACHE_SIZE: u32 = 16;
    let size = std::env::var("HELMER_MESHOPT_CACHE_SIZE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_CACHE_SIZE);
    size.clamp(3, 64)
}

fn meshopt_simplify_options() -> SimplifyOptions {
    // Keep options conservative and deterministic across platforms
    SimplifyOptions::LockBorder | SimplifyOptions::Regularize
}

fn meshopt_simplify_kind() -> MeshoptSimplifyKind {
    std::env::var("HELMER_MESHOPT_SIMPLIFY_KIND")
        .ok()
        .map(|v| v.to_lowercase())
        .as_deref()
        .map(|v| match v {
            "standard" => MeshoptSimplifyKind::Standard,
            "sloppy" => MeshoptSimplifyKind::Sloppy,
            _ => MeshoptSimplifyKind::Sloppy,
        })
        .unwrap_or(MeshoptSimplifyKind::Sloppy)
}

fn meshopt_simplify_indices(
    indices: &[u32],
    adapter: &VertexDataAdapter<'_>,
    target_index_count: usize,
    simplification_error: f32,
) -> Vec<u32> {
    #[cfg(target_arch = "wasm32")]
    {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || match meshopt_simplify_kind() {
                    MeshoptSimplifyKind::Standard => simplify(
                        indices,
                        adapter,
                        target_index_count,
                        simplification_error,
                        meshopt_simplify_options(),
                        None,
                    ),
                    MeshoptSimplifyKind::Sloppy => simplify_sloppy(
                        indices,
                        adapter,
                        target_index_count,
                        simplification_error,
                        None,
                    ),
                },
            ));
        match result {
            Ok(data) => data,
            Err(_) => {
                disable_meshopt_runtime("mesh simplification panicked");
                Vec::new()
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        match meshopt_simplify_kind() {
            MeshoptSimplifyKind::Standard => simplify(
                indices,
                adapter,
                target_index_count,
                simplification_error,
                meshopt_simplify_options(),
                None,
            ),
            MeshoptSimplifyKind::Sloppy => simplify_sloppy(
                indices,
                adapter,
                target_index_count,
                simplification_error,
                None,
            ),
        }
    }
}

fn meshopt_run_vertex_cache(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    #[cfg(target_arch = "wasm32")]
    {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match meshopt_cache_algo() {
                MeshoptCacheAlgo::Forsyth => optimize_vertex_cache(indices, vertex_count),
                MeshoptCacheAlgo::Fifo => {
                    optimize_vertex_cache_fifo(indices, vertex_count, meshopt_cache_size())
                }
            }
        }));
        match result {
            Ok(data) => data,
            Err(_) => {
                disable_meshopt_runtime("vertex cache optimization panicked");
                Vec::new()
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        match meshopt_cache_algo() {
            MeshoptCacheAlgo::Forsyth => optimize_vertex_cache(indices, vertex_count),
            MeshoptCacheAlgo::Fifo => {
                optimize_vertex_cache_fifo(indices, vertex_count, meshopt_cache_size())
            }
        }
    }
}

fn meshopt_indices_valid(indices: &[u32], vertex_count: usize) -> bool {
    if indices.len() < 3 || indices.len() % 3 != 0 {
        return false;
    }
    let Some(max_index) = indices.iter().copied().max() else {
        return false;
    };
    (max_index as usize) < vertex_count
}

fn meshopt_working_set_bytes(vertex_count: usize, index_count: usize) -> usize {
    let face_count = index_count / 3;
    let vertex_bytes = vertex_count.saturating_mul(12);
    let face_bytes = face_count.saturating_mul(17);
    vertex_bytes.saturating_add(face_bytes)
}

#[derive(Debug)]
struct MeshoptPrepared {
    global_indices: Vec<u32>,
    local_indices: Vec<u32>,
    local_to_global: Vec<u32>,
    dropped_invalid: usize,
    dropped_degenerate: usize,
    dropped_nonfinite: usize,
    dropped_trailing: usize,
}

impl MeshoptPrepared {
    fn dropped_total(&self) -> usize {
        self.dropped_invalid + self.dropped_degenerate + self.dropped_nonfinite
    }

    fn local_vertex_count(&self) -> usize {
        self.local_to_global.len()
    }

    fn map_to_global(&self, local_indices: &[u32]) -> Vec<u32> {
        if local_indices.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(local_indices.len());
        for &idx in local_indices {
            match self.local_to_global.get(idx as usize) {
                Some(&global) => out.push(global),
                None => {
                    warn!(
                        "meshopt: local index {} out of range ({}).",
                        idx,
                        self.local_to_global.len()
                    );
                    return self.global_indices.clone();
                }
            }
        }
        out
    }
}

fn meshopt_positions_finite(position: [f32; 3]) -> bool {
    position[0].is_finite() && position[1].is_finite() && position[2].is_finite()
}

fn meshopt_prepare_mesh(vertices: &[Vertex], indices: &[u32]) -> Option<MeshoptPrepared> {
    if vertices.is_empty() || indices.len() < 3 {
        return None;
    }
    if vertices.len() > u32::MAX as usize {
        warn!("meshopt: vertex count exceeds u32::MAX; skipping.");
        return None;
    }

    let vertex_count = vertices.len();
    let vertex_ok: Vec<bool> = vertices
        .iter()
        .map(|v| meshopt_positions_finite(v.position))
        .collect();

    let mut remap = vec![u32::MAX; vertex_count];
    let mut local_to_global = Vec::new();
    let mut local_indices = Vec::with_capacity(indices.len());
    let mut global_indices = Vec::with_capacity(indices.len());

    let mut dropped_invalid = 0usize;
    let mut dropped_degenerate = 0usize;
    let mut dropped_nonfinite = 0usize;
    let dropped_trailing = indices.len() % 3;

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        if i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count {
            dropped_invalid += 1;
            continue;
        }
        if i0 == i1 || i1 == i2 || i0 == i2 {
            dropped_degenerate += 1;
            continue;
        }
        if !vertex_ok[i0] || !vertex_ok[i1] || !vertex_ok[i2] {
            dropped_nonfinite += 1;
            continue;
        }

        global_indices.extend_from_slice(tri);
        for &idx in tri {
            let entry = &mut remap[idx as usize];
            let local = if *entry == u32::MAX {
                let next = local_to_global.len() as u32;
                *entry = next;
                local_to_global.push(idx);
                next
            } else {
                *entry
            };
            local_indices.push(local);
        }
    }

    if global_indices.is_empty() || local_indices.is_empty() {
        return None;
    }
    if local_to_global.len() > u32::MAX as usize {
        warn!("meshopt: remapped vertex count exceeds u32::MAX; skipping.");
        return None;
    }

    Some(MeshoptPrepared {
        global_indices,
        local_indices,
        local_to_global,
        dropped_invalid,
        dropped_degenerate,
        dropped_nonfinite,
        dropped_trailing,
    })
}

fn fallback_render_indices(vertex_count: usize, indices: &[u32]) -> Option<Vec<u32>> {
    if vertex_count == 0 || indices.len() < 3 {
        return None;
    }
    let trimmed = indices.len() - (indices.len() % 3);
    if trimmed < 3 {
        return None;
    }
    let slice = &indices[..trimmed];
    let max_index = slice.iter().copied().max().unwrap_or(0) as usize;
    if max_index >= vertex_count {
        warn!(
            "meshopt: fallback indices out of range (max {}, vertex_count {}).",
            max_index, vertex_count
        );
        return None;
    }
    if trimmed != indices.len() {
        warn!(
            "meshopt: trimming {} trailing indices for fallback.",
            indices.len() - trimmed
        );
    }
    Some(slice.to_vec())
}

fn build_meshopt_chunk(
    indices: &[u32],
    start: usize,
    budget_bytes: usize,
) -> (Vec<u32>, Vec<u32>, usize) {
    let mut remap: HashMap<u32, u32> = HashMap::new();
    let mut local_to_global: Vec<u32> = Vec::new();
    let mut local_indices: Vec<u32> = Vec::new();
    let mut face_count = 0usize;
    let mut cursor = start;

    while cursor + 2 < indices.len() {
        let tri = &indices[cursor..cursor + 3];
        let mut new_unique = 0usize;
        for &idx in tri {
            if !remap.contains_key(&idx) {
                new_unique += 1;
            }
        }

        let prospective_vertices = local_to_global.len() + new_unique;
        let prospective_faces = face_count + 1;
        if face_count > 0
            && meshopt_working_set_bytes(prospective_vertices, prospective_faces * 3) > budget_bytes
        {
            break;
        }

        for &idx in tri {
            let local = *remap.entry(idx).or_insert_with(|| {
                local_to_global.push(idx);
                (local_to_global.len() - 1) as u32
            });
            local_indices.push(local);
        }

        face_count += 1;
        cursor += 3;
    }

    (local_indices, local_to_global, cursor)
}

fn optimize_vertex_cache_chunked(indices: &[u32], budget_bytes: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(indices.len());
    let mut cursor = 0usize;

    while cursor + 2 < indices.len() {
        let (local_indices, local_to_global, next_cursor) =
            build_meshopt_chunk(indices, cursor, budget_bytes);
        if local_indices.is_empty() {
            break;
        }
        let optimized_local = meshopt_run_vertex_cache(&local_indices, local_to_global.len());
        if !meshopt_indices_valid(&optimized_local, local_to_global.len()) {
            #[cfg(target_arch = "wasm32")]
            {
                disable_meshopt_runtime("vertex cache optimization returned invalid indices");
            }
            out.extend(
                local_indices
                    .into_iter()
                    .map(|idx| local_to_global[idx as usize]),
            );
            cursor = next_cursor;
            continue;
        }
        out.extend(
            optimized_local
                .into_iter()
                .map(|idx| local_to_global[idx as usize]),
        );
        cursor = next_cursor;
    }

    out
}

fn meshopt_optimize_local_indices(
    local_indices: &[u32],
    local_vertex_count: usize,
    budget_bytes: usize,
) -> Vec<u32> {
    if local_indices.is_empty() {
        return Vec::new();
    }
    if !meshopt_vertex_cache_enabled() {
        return local_indices.to_vec();
    }
    if !meshopt_can_run(local_vertex_count, local_indices, None) {
        return local_indices.to_vec();
    }

    let optimized =
        if meshopt_working_set_bytes(local_vertex_count, local_indices.len()) <= budget_bytes {
            meshopt_run_vertex_cache(local_indices, local_vertex_count)
        } else {
            optimize_vertex_cache_chunked(local_indices, budget_bytes)
        };

    if meshopt_indices_valid(&optimized, local_vertex_count) {
        optimized
    } else {
        #[cfg(target_arch = "wasm32")]
        {
            disable_meshopt_runtime("vertex cache optimization returned invalid indices");
        }
        warn!(
            "meshopt: optimized indices invalid (len {}, vertices {}); falling back.",
            optimized.len(),
            local_vertex_count
        );
        local_indices.to_vec()
    }
}

fn simplify_chunked(
    positions: &[[f32; 3]],
    indices: &[u32],
    ratio: f32,
    simplification_error: f32,
    budget_bytes: usize,
) -> Vec<u32> {
    let mut out = Vec::new();
    let mut cursor = 0usize;

    while cursor + 2 < indices.len() {
        let (local_indices, local_to_global, next_cursor) =
            build_meshopt_chunk(indices, cursor, budget_bytes);
        if local_indices.is_empty() {
            break;
        }
        let mut local_positions: Vec<[f32; 3]> = Vec::with_capacity(local_to_global.len());
        let mut valid = true;
        for &global_idx in &local_to_global {
            if let Some(position) = positions.get(global_idx as usize) {
                local_positions.push(*position);
            } else {
                valid = false;
                break;
            }
        }
        if !valid {
            out.extend(
                local_indices
                    .into_iter()
                    .map(|idx| local_to_global[idx as usize]),
            );
            cursor = next_cursor;
            continue;
        }

        let adapter = VertexDataAdapter::new(
            bytemuck::cast_slice(&local_positions),
            std::mem::size_of::<[f32; 3]>(),
            0,
        );

        let target_index_count = ((local_indices.len() as f32) * ratio) as usize;
        let target_index_count = target_index_count - (target_index_count % 3);

        let simplified_local = if let Ok(adapter) = adapter {
            if target_index_count >= 3 && target_index_count < local_indices.len() {
                meshopt_simplify_indices(
                    &local_indices,
                    &adapter,
                    target_index_count,
                    simplification_error,
                )
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let simplified_local = if meshopt_indices_valid(&simplified_local, local_to_global.len()) {
            simplified_local
        } else {
            if !simplified_local.is_empty() {
                #[cfg(target_arch = "wasm32")]
                {
                    disable_meshopt_runtime("mesh simplification returned invalid indices");
                }
                warn!(
                    "meshopt simplify: invalid indices (len {}, vertices {}); falling back.",
                    simplified_local.len(),
                    local_to_global.len()
                );
            }
            Vec::new()
        };

        let result_local = if simplified_local.is_empty() {
            local_indices
        } else {
            simplified_local
        };
        out.extend(
            result_local
                .into_iter()
                .map(|idx| local_to_global[idx as usize]),
        );
        cursor = next_cursor;
    }

    out
}

fn meshopt_limits(tuning: Option<&AssetStreamingTuning>) -> (usize, usize) {
    let tuning = tuning.copied().unwrap_or_default();
    let vertex_limit = tuning.lod_safe_vertex_limit;
    let index_limit = tuning.lod_safe_index_limit;

    (vertex_limit, index_limit)
}

fn meshopt_can_run(
    vertex_count: usize,
    indices: &[u32],
    tuning: Option<&AssetStreamingTuning>,
) -> bool {
    if vertex_count == 0 || indices.is_empty() {
        return false;
    }
    if indices.len() % 3 != 0 {
        return false;
    }
    if vertex_count > u32::MAX as usize {
        return false;
    }
    let (vertex_limit, index_limit) = meshopt_limits(tuning);
    if vertex_count > vertex_limit || indices.len() > index_limit {
        return false;
    }
    let max_index = match indices.iter().copied().max() {
        Some(value) => value as usize,
        None => return false,
    };
    max_index < vertex_count
}

fn maybe_optimize_vertex_cache(vertices: &[Vertex], indices: &[u32]) -> Vec<u32> {
    if !meshopt_vertex_cache_enabled() {
        return fallback_render_indices(vertices.len(), indices)
            .unwrap_or_else(|| indices.to_vec());
    }
    let fallback = fallback_render_indices(vertices.len(), indices);
    let prepared = match meshopt_prepare_mesh(vertices, indices) {
        Some(prepared) => prepared,
        None => {
            if fallback.is_none() {
                warn!("meshopt: unable to prepare mesh; falling back to empty indices.");
            }
            return fallback.unwrap_or_default();
        }
    };

    if prepared.dropped_trailing > 0 {
        warn!(
            "meshopt: dropped {} trailing index values (index count not divisible by 3).",
            prepared.dropped_trailing
        );
    }
    let dropped = prepared.dropped_total();
    if dropped > 0 {
        warn!(
            "meshopt: dropped {} invalid/degenerate triangles out of {}.",
            dropped,
            (indices.len() / 3) + dropped
        );
    }

    let local_vertex_count = prepared.local_vertex_count();
    if !meshopt_can_run(local_vertex_count, &prepared.local_indices, None) {
        return if prepared.global_indices.is_empty() {
            fallback.unwrap_or_default()
        } else {
            prepared.global_indices
        };
    }

    let budget = meshopt_budget_bytes();
    let optimized_local =
        meshopt_optimize_local_indices(&prepared.local_indices, local_vertex_count, budget);
    let mapped = prepared.map_to_global(&optimized_local);
    if mapped.is_empty() {
        return fallback.unwrap_or_default();
    }
    if meshopt_indices_valid(&mapped, vertices.len()) {
        mapped
    } else {
        warn!(
            "meshopt: invalid cached indices (len {}, vertices {}); falling back.",
            mapped.len(),
            vertices.len()
        );
        fallback.unwrap_or_else(|| prepared.global_indices)
    }
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
        maybe_optimize_vertex_cache(vertices, indices)
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

    let fallback = fallback_render_indices(vertices.len(), indices);
    let prepared = match meshopt_prepare_mesh(vertices, indices) {
        Some(prepared) => prepared,
        None => {
            if let Some(fallback) = fallback {
                lods.push(fallback);
            } else {
                warn!("meshopt: no valid indices for LOD generation.");
            }
            return lods;
        }
    };
    if prepared.dropped_trailing > 0 {
        warn!(
            "meshopt: dropped {} trailing index values (index count not divisible by 3).",
            prepared.dropped_trailing
        );
    }
    let dropped = prepared.dropped_total();
    if dropped > 0 {
        warn!(
            "meshopt simplify: dropped {} invalid/degenerate triangles out of {}.",
            dropped,
            (indices.len() / 3) + dropped
        );
    }
    let indices = &prepared.global_indices;
    if indices.is_empty() {
        if let Some(fallback) = fallback {
            lods.push(fallback);
        } else {
            warn!("meshopt: all triangles dropped; no LODs produced.");
        }
        return lods;
    }

    // Allow opting out of meshopt paths entirely for robustness.
    if !meshopt_enabled() {
        lods.push(fallback.unwrap_or_else(|| indices.clone()));
        return lods;
    }

    let local_vertex_count = prepared.local_vertex_count();
    if !meshopt_can_run(local_vertex_count, &prepared.local_indices, Some(tuning)) {
        lods.push(indices.clone());
        return lods;
    }

    let budget = meshopt_budget_bytes();
    // LOD 0: The original mesh, optimized for the GPU vertex cache.
    let lod0_local =
        meshopt_optimize_local_indices(&prepared.local_indices, local_vertex_count, budget);
    let lod0 = prepared.map_to_global(&lod0_local);
    let lod0 = if lod0.is_empty() {
        indices.clone()
    } else if meshopt_indices_valid(&lod0, vertices.len()) {
        lod0
    } else {
        warn!(
            "meshopt: LOD0 indices invalid (len {}, vertices {}); using original.",
            lod0.len(),
            vertices.len()
        );
        indices.clone()
    };
    lods.push(lod0);

    if !meshopt_simplify_enabled() {
        return lods;
    }

    // Prepare the vertex data adapter for the simplifier.
    // The adapter tells meshopt how to access the position data from your Vertex struct.
    let positions: Vec<[f32; 3]> = prepared
        .local_to_global
        .iter()
        .map(|&idx| vertices[idx as usize].position)
        .collect();

    let vertex_data_adapter = match VertexDataAdapter::new(
        bytemuck::cast_slice(&positions),
        std::mem::size_of::<[f32; 3]>(),
        0,
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
        if target_index_count < 3 {
            continue;
        }

        if target_index_count < indices.len() {
            // Always simplify from the original, high-quality index buffer.
            let simplified_local =
                if meshopt_working_set_bytes(local_vertex_count, prepared.local_indices.len())
                    <= budget
                {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        meshopt_simplify_indices(
                            &prepared.local_indices,
                            &vertex_data_adapter,
                            target_index_count,
                            simplification_error,
                        )
                    })) {
                        Ok(data) => data,
                        Err(_) => {
                            warn!("meshopt simplification panicked; keeping previous LODs only.");
                            break;
                        }
                    }
                } else {
                    simplify_chunked(
                        &positions,
                        &prepared.local_indices,
                        ratio,
                        simplification_error,
                        budget,
                    )
                };

            if !simplified_local.is_empty() {
                let optimized_local =
                    meshopt_optimize_local_indices(&simplified_local, local_vertex_count, budget);
                let simplified_global = prepared.map_to_global(&optimized_local);
                if simplified_global.is_empty() {
                    if let Some(prev_lod) = lods.last() {
                        lods.push(prev_lod.clone());
                    }
                } else if meshopt_indices_valid(&simplified_global, vertices.len()) {
                    lods.push(simplified_global);
                } else if let Some(prev_lod) = lods.last() {
                    warn!(
                        "meshopt: simplified indices invalid (len {}, vertices {}); reusing previous LOD.",
                        simplified_global.len(),
                        vertices.len()
                    );
                    lods.push(prev_lod.clone());
                }
            } else if let Some(prev_lod) = lods.last() {
                // If simplification fails (e.g., target is too low), reuse the previous LOD.
                lods.push(prev_lod.clone());
            }
        }
    }
    lods
}

pub(crate) struct MeshPayload {
    pub(crate) lods: Vec<MeshLodPayload>,
    pub(crate) total_lods: usize,
    pub(crate) bounds: Aabb,
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

pub(crate) fn build_mesh_payload(
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
        warn!("meshopt produced no valid LODs; falling back to original mesh.");
        if let Some(lod) = compact_mesh_lod(&vertices, &indices, 0) {
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

pub(crate) fn estimate_primitive_bounds(
    doc: &gltf::Document,
    desc: &MeshPrimitiveDesc,
) -> Option<Aabb> {
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

pub(crate) fn process_primitive<B: BufferSource>(
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
    let mut joints_iter = reader.read_joints(0).map(|j| j.into_u16());
    let mut weights_iter = reader.read_weights(0).map(|w| w.into_f32());

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
        let joints = joints_iter
            .as_mut()
            .and_then(|iter| iter.next())
            .map(|joint| {
                [
                    joint[0] as u32,
                    joint[1] as u32,
                    joint[2] as u32,
                    joint[3] as u32,
                ]
            })
            .unwrap_or([0; 4]);
        let weights = weights_iter
            .as_mut()
            .and_then(|iter| iter.next())
            .map(|weight| {
                let sum = weight[0] + weight[1] + weight[2] + weight[3];
                if sum > 0.0 {
                    [
                        weight[0] / sum,
                        weight[1] / sum,
                        weight[2] / sum,
                        weight[3] / sum,
                    ]
                } else {
                    [1.0, 0.0, 0.0, 0.0]
                }
            })
            .unwrap_or([1.0, 0.0, 0.0, 0.0]);
        vertices.push(Vertex::with_skinning(
            position, normal, tex_coord, [0.0; 4], joints, weights,
        ));
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

pub(crate) fn parse_glb(bytes: &[u8]) -> Result<(Vec<Vertex>, Vec<u32>, Aabb), String> {
    let (gltf, buffers, _) = gltf::import_slice(bytes).map_err(|e| e.to_string())?;
    parse_mesh_document(&gltf, &buffers)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_scene_glb_path(path: &Path) -> Result<ParsedGltfScene, String> {
    let gltf = gltf::Gltf::open(path).map_err(|e| e.to_string())?;
    parse_scene_document(gltf.document, None, path.parent(), Some(path.to_path_buf()))
}

pub(crate) fn parse_mesh_document<B: BufferSource>(
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

pub(crate) fn parse_scene_document(
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
        alpha_mode: AlphaMode::Opaque,
        alpha_cutoff: None,
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
        let alpha_mode = match mat.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        };
        let alpha_cutoff = if matches!(alpha_mode, AlphaMode::Mask) {
            mat.alpha_cutoff()
        } else {
            None
        };

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
            alpha_mode,
            alpha_cutoff,
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
                    node_index: node.index(),
                    skin_index: node.skin().map(|skin| skin.index()),
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

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
struct ParsedAnimationChannel {
    node_index: usize,
    channel: AnimationChannel,
}

#[cfg(not(target_arch = "wasm32"))]
fn build_parent_map(doc: &gltf::Document) -> Vec<Option<usize>> {
    let mut parents = vec![None; doc.nodes().len()];
    for scene in doc.scenes() {
        for node in scene.nodes() {
            traverse_node(&node, None, &mut parents);
        }
    }
    parents
}

#[cfg(not(target_arch = "wasm32"))]
fn traverse_node(node: &gltf::Node, parent: Option<usize>, parents: &mut [Option<usize>]) {
    let idx = node.index();
    if idx < parents.len() {
        parents[idx] = parent;
    }
    for child in node.children() {
        traverse_node(&child, Some(idx), parents);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_animation_channels(
    animation: &gltf::Animation,
    buffers: &[StreamedBuffer],
) -> Vec<ParsedAnimationChannel> {
    let mut channels = Vec::new();
    for channel in animation.channels() {
        let node_index = channel.target().node().index();
        let sampler = channel.sampler();
        let interpolation = map_interpolation(sampler.interpolation());
        let reader = channel.reader(|buffer| buffers.get(buffer.index()).map(|b| b.as_slice()));
        let Some(inputs) = reader.read_inputs().map(|iter| iter.collect::<Vec<f32>>()) else {
            continue;
        };
        let Some(outputs) = reader.read_outputs() else {
            continue;
        };
        match outputs {
            ReadOutputs::Translations(translations) => {
                let outputs: Vec<[f32; 3]> = translations.collect();
                let keyframes = parse_vec3_keyframes(&inputs, &outputs, interpolation);
                channels.push(ParsedAnimationChannel {
                    node_index,
                    channel: AnimationChannel::Translation {
                        target: node_index,
                        interpolation,
                        keyframes,
                    },
                });
            }
            ReadOutputs::Scales(scales) => {
                let outputs: Vec<[f32; 3]> = scales.collect();
                let keyframes = parse_vec3_keyframes(&inputs, &outputs, interpolation);
                channels.push(ParsedAnimationChannel {
                    node_index,
                    channel: AnimationChannel::Scale {
                        target: node_index,
                        interpolation,
                        keyframes,
                    },
                });
            }
            ReadOutputs::Rotations(rotations) => {
                let outputs: Vec<[f32; 4]> = rotations.into_f32().collect();
                let keyframes = parse_quat_keyframes(&inputs, &outputs, interpolation);
                channels.push(ParsedAnimationChannel {
                    node_index,
                    channel: AnimationChannel::Rotation {
                        target: node_index,
                        interpolation,
                        keyframes,
                    },
                });
            }
            ReadOutputs::MorphTargetWeights(_) => {}
        }
    }
    channels
}

#[cfg(not(target_arch = "wasm32"))]
fn map_interpolation(interp: GltfInterpolation) -> AnimInterp {
    match interp {
        GltfInterpolation::Step => AnimInterp::Step,
        GltfInterpolation::Linear => AnimInterp::Linear,
        GltfInterpolation::CubicSpline => AnimInterp::Cubic,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_vec3_keyframes(
    inputs: &[f32],
    outputs: &[[f32; 3]],
    interpolation: AnimInterp,
) -> Vec<Keyframe<Vec3>> {
    let mut keyframes = Vec::new();
    if inputs.is_empty() {
        return keyframes;
    }
    if interpolation == AnimInterp::Cubic && outputs.len() >= inputs.len() * 3 {
        for (idx, time) in inputs.iter().enumerate() {
            let base = idx * 3;
            let in_tangent = Vec3::from(outputs[base]);
            let value = Vec3::from(outputs[base + 1]);
            let out_tangent = Vec3::from(outputs[base + 2]);
            keyframes.push(Keyframe {
                time: *time,
                value,
                in_tangent: Some(in_tangent),
                out_tangent: Some(out_tangent),
            });
        }
        return keyframes;
    }
    for (time, value) in inputs.iter().zip(outputs.iter()) {
        keyframes.push(Keyframe::new(*time, Vec3::from(*value)));
    }
    keyframes
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_quat_keyframes(
    inputs: &[f32],
    outputs: &[[f32; 4]],
    interpolation: AnimInterp,
) -> Vec<Keyframe<Quat>> {
    let mut keyframes = Vec::new();
    if inputs.is_empty() {
        return keyframes;
    }
    if interpolation == AnimInterp::Cubic && outputs.len() >= inputs.len() * 3 {
        for (idx, time) in inputs.iter().enumerate() {
            let base = idx * 3;
            let in_tangent = Quat::from_array(outputs[base]).normalize();
            let value = Quat::from_array(outputs[base + 1]).normalize();
            let out_tangent = Quat::from_array(outputs[base + 2]).normalize();
            keyframes.push(Keyframe {
                time: *time,
                value,
                in_tangent: Some(in_tangent),
                out_tangent: Some(out_tangent),
            });
        }
        return keyframes;
    }
    for (time, value) in inputs.iter().zip(outputs.iter()) {
        keyframes.push(Keyframe::new(*time, Quat::from_array(*value).normalize()));
    }
    keyframes
}

#[cfg(not(target_arch = "wasm32"))]
fn retarget_channel(channel: &mut AnimationChannel, target: usize) {
    match channel {
        AnimationChannel::Translation { target: t, .. } => *t = target,
        AnimationChannel::Rotation { target: t, .. } => *t = target,
        AnimationChannel::Scale { target: t, .. } => *t = target,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn channel_duration(channel: &AnimationChannel) -> f32 {
    match channel {
        AnimationChannel::Translation { keyframes, .. }
        | AnimationChannel::Scale { keyframes, .. } => {
            keyframes.last().map(|k| k.time).unwrap_or(0.0)
        }
        AnimationChannel::Rotation { keyframes, .. } => {
            keyframes.last().map(|k| k.time).unwrap_or(0.0)
        }
    }
}

#[cfg(any())]
async fn handle_request_web(request: AssetLoadRequest, io: WebAssetIo) -> Option<AssetLoadResult> {
    let path_str = request_path(&request).to_string_lossy().to_string();
    match request {
        AssetLoadRequest::Mesh { id, path, tuning } => {
            let parsed = if is_gltf_path(&path) {
                match load_gltf_streaming_web(&path, &io).await {
                    Ok((doc, buffers)) => match parse_mesh_document(&doc, &buffers) {
                        Ok(data) => Some(data),
                        Err(e) => {
                            warn!("Failed to parse glTF mesh '{}': {}", path_str, e);
                            None
                        }
                    },
                    Err(e) => {
                        warn!("Failed to parse glTF mesh '{}': {}", path_str, e);
                        None
                    }
                }
            } else {
                match io.read_path(&path).await {
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
                build_mesh_payload(vertices, indices, bounds, &tuning).map(|payload| {
                    AssetLoadResult::Mesh {
                        id,
                        scene_id: None,
                        lods: payload.lods,
                        total_lods: payload.total_lods,
                        bounds: payload.bounds,
                    }
                })
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
        AssetLoadRequest::Texture { id, path, kind } => {
            load_and_parse_web(
                id,
                &path,
                &io,
                |bytes| decode_texture_file_bytes(kind, bytes, &path),
                |id, data| AssetLoadResult::Texture {
                    id,
                    scene_id: None,
                    name: path_str.clone(),
                    kind,
                    data,
                },
            )
            .await
        }
        AssetLoadRequest::Material { id, path } => {
            load_and_parse_web(id, &path, &io, parse_ron_material, |id, data| {
                AssetLoadResult::Material { id, data }
            })
            .await
        }
        AssetLoadRequest::Scene { id, path } => {
            if is_gltf_path(&path) {
                match load_gltf_streaming_web(&path, &io).await {
                    Ok((doc, buffers)) => match parse_scene_document(
                        doc,
                        Some(buffers),
                        path.parent(),
                        Some(path.to_path_buf()),
                    ) {
                        Ok(data) => Some(AssetLoadResult::Scene { id, data }),
                        Err(e) => {
                            warn!("Failed to parse glTF scene '{}': {}", path_str, e);
                            None
                        }
                    },
                    Err(e) => {
                        warn!("Failed to parse glTF scene '{}': {}", path_str, e);
                        None
                    }
                }
            } else {
                match io.read_path(&path).await {
                    Ok(bytes) => match gltf::import_slice(&bytes) {
                        Ok((doc, buffers, _)) => {
                            let buffers = buffers
                                .into_iter()
                                .map(|data| StreamedBuffer::owned(data.to_vec()))
                                .collect();
                            match parse_scene_document(
                                doc,
                                Some(buffers),
                                path.parent(),
                                Some(path.to_path_buf()),
                            ) {
                                Ok(data) => Some(AssetLoadResult::Scene { id, data }),
                                Err(e) => {
                                    warn!("Failed to parse glTF scene '{}': {}", path_str, e);
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to parse glTF scene '{}': {}", path_str, e);
                            None
                        }
                    },
                    Err(e) => {
                        warn!("Failed to read scene file '{}': {}", path_str, e);
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
        } => match load_scene_buffers_web(&doc, scene_path.as_deref(), base_path.as_deref(), &io)
            .await
        {
            Ok(buffers) => {
                let buffers_bytes: usize = buffers.iter().map(|buffer| buffer.len()).sum();
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
                    build_mesh_payload(vertices, indices, bounds, &tuning).map_or_else(
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
                let decoded = decode_texture_asset_web(
                    gltf_tex,
                    buffers.buffers(),
                    base_path.as_deref(),
                    &io,
                    kind,
                )
                .await;
                decoded
                    .map(
                        |(name, kind, data, format, dimensions)| AssetLoadResult::Texture {
                            id,
                            scene_id: Some(scene_id),
                            name,
                            kind,
                            data: (data, format, dimensions),
                        },
                    )
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
            let low = generate_low_res_from_parts(data.as_ref(), format, dimensions, max_dim);
            Some(AssetLoadResult::LowResTexture {
                id,
                name,
                kind,
                data: low,
            })
        }
    }
}
