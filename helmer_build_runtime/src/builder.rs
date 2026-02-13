use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{Seek, Write},
    ops::Range,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crc32fast::Hasher as Crc32;
use flate2::{Compression, write::ZlibEncoder};
use helmer_editor_runtime::{
    project::{ProjectConfig, ProjectDescriptor, normalize_asset_relative_path},
    scripting::{
        is_script_path, resolve_rust_script_manifest, rust_dynamic_library_extension,
        rust_dynamic_library_prefix, rust_manifest_names, rust_prebuilt_plugin_relative_path,
    },
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::{
    crypto::{
        fnv1a64, hex_encode, key_fingerprint, parse_key_material, path_stream_id,
        payload_stream_id, transform_in_place,
    },
    format::{
        AssetKind, PACK_FLAG_CHUNK_DEDUP, PACK_FLAG_COMPRESSED, PACK_FLAG_PATHS_TRANSFORMED,
        PACK_FLAG_TRANSFORMED, PACK_HEADER_SIZE, PACK_SET_MANIFEST_VERSION, PACK_VERSION,
        PackEntry, PackHeader, PackSetManifest, PackSetPack, PackToc,
    },
};

const CHUNK_ALIGN: u64 = 16;
const DEFAULT_MAX_PACK_BYTES: u64 = 512 * 1024 * 1024;
const DEFAULT_MAX_ASSETS_PER_PACK: usize = 50_000;
const ASSET_ENTRY_OVERHEAD_ESTIMATE: u64 = 128;
const EMBEDDED_SCRIPT_SDK_CARGO_TOML: &str = include_str!("../../helmer_script_sdk/Cargo.toml");
const EMBEDDED_SCRIPT_SDK_LIB_RS: &str = include_str!("../../helmer_script_sdk/src/lib.rs");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildRequest {
    pub project_root: String,
    pub output_pack: String,
    pub key: String,
    pub compression_level: Option<u32>,
    pub include_hidden: Option<bool>,
    pub max_threads: Option<usize>,
    pub max_pack_bytes: Option<u64>,
    pub max_assets_per_pack: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildPackOutput {
    pub pack_path: String,
    pub asset_count: usize,
    pub chunk_count: usize,
    pub deduped_assets: usize,
    pub bytes_written: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildResponse {
    pub pack_path: String,
    pub pack_paths: Vec<String>,
    pub manifest_path: String,
    pub pack_count: usize,
    pub packs: Vec<BuildPackOutput>,
    pub project_root: String,
    pub project_name: String,
    pub project_config: ProjectConfig,
    pub startup_scene: String,
    pub created_unix_ms: u64,
    pub bytes_written: u64,
    pub asset_count: usize,
    pub chunk_count: usize,
    pub deduped_assets: usize,
    pub elapsed_ms: u128,
    pub key_fingerprint: String,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct PendingAsset {
    source_path: PathBuf,
    relative_path: String,
    asset_kind: AssetKind,
    source_flags: u32,
    source_mtime_unix_ms: i64,
}

#[derive(Debug, Clone)]
struct ProcessedAsset {
    relative_path: String,
    asset_id: u64,
    asset_kind: AssetKind,
    source_flags: u32,
    source_mtime_unix_ms: i64,
    source_len: u64,
    source_crc32: u32,
    source_hash: u64,
    compressed_len: u64,
    chunk_payload: Vec<u8>,
    chunk_crc32: u32,
    chunk_hash: u64,
    path_blob: Vec<u8>,
}

#[derive(Debug, Clone)]
struct PackWriteStats {
    pack_path: PathBuf,
    asset_count: usize,
    chunk_count: usize,
    deduped_assets: usize,
    bytes_written: u64,
}

pub fn build_project_pack(request: &BuildRequest) -> Result<BuildResponse, String> {
    let started = Instant::now();

    if let Some(threads) = request.max_threads {
        if threads > 0 {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global();
        }
    }

    let descriptor = ProjectDescriptor::from_root(&request.project_root)?;
    let include_hidden = request.include_hidden.unwrap_or(false);
    let compression_level = request.compression_level.unwrap_or(6).min(9);
    let max_pack_bytes = request
        .max_pack_bytes
        .unwrap_or(DEFAULT_MAX_PACK_BYTES)
        .max(1);
    let max_assets_per_pack = request
        .max_assets_per_pack
        .unwrap_or(DEFAULT_MAX_ASSETS_PER_PACK)
        .max(1);

    let key_material = parse_key_material(&request.key)?;
    let key_fingerprint_bytes = key_fingerprint(&key_material);

    let mut warnings = Vec::new();
    let startup_scene = resolve_startup_scene(&descriptor, &mut warnings)?;
    let mut assets = collect_assets(&descriptor, include_hidden)?;
    if assets.is_empty() {
        warnings.push("No assets discovered under project assets root".to_string());
    }
    let output_path = PathBuf::from(&request.output_pack);
    let prebuilt_rust_assets =
        build_precompiled_rust_script_assets(&descriptor, &assets, &output_path, &mut warnings)?;
    assets.extend(prebuilt_rust_assets);
    let mut deduped_assets = HashMap::with_capacity(assets.len());
    for asset in assets {
        deduped_assets.insert(asset.relative_path.clone(), asset);
    }
    let mut assets = deduped_assets.into_values().collect::<Vec<_>>();

    assets.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    let key_ref = key_material.as_slice();
    let processed_results: Vec<Result<ProcessedAsset, String>> = assets
        .par_iter()
        .map(|asset| process_asset(asset, key_ref, compression_level))
        .collect();

    let mut processed = Vec::with_capacity(processed_results.len());
    for result in processed_results {
        processed.push(result?);
    }
    processed.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    let shard_ranges = shard_assets(&processed, max_pack_bytes, max_assets_per_pack);
    let output_paths = output_paths_for_shards(&output_path, shard_ranges.len());

    let created_unix_ms = now_unix_ms();
    let mut pack_stats = Vec::with_capacity(shard_ranges.len());
    for (index, range) in shard_ranges.iter().enumerate() {
        let stats = write_single_pack(
            &output_paths[index],
            &processed[range.clone()],
            created_unix_ms,
            key_fingerprint_bytes,
        )?;
        pack_stats.push(stats);
    }

    if pack_stats.len() > 1 {
        warnings.push(format!(
            "Pack set split into {} packs (max_pack_bytes={}, max_assets_per_pack={})",
            pack_stats.len(),
            max_pack_bytes,
            max_assets_per_pack
        ));
    }

    let manifest_path = manifest_path_for_output(&output_path);
    let manifest = build_pack_manifest(
        &descriptor,
        &output_path,
        &manifest_path,
        &pack_stats,
        created_unix_ms,
        &key_fingerprint_bytes,
    );
    write_pack_manifest(&manifest_path, &manifest)?;

    let packs: Vec<BuildPackOutput> = pack_stats
        .iter()
        .map(|stats| BuildPackOutput {
            pack_path: stats.pack_path.to_string_lossy().into_owned(),
            asset_count: stats.asset_count,
            chunk_count: stats.chunk_count,
            deduped_assets: stats.deduped_assets,
            bytes_written: stats.bytes_written,
        })
        .collect();

    let pack_paths = packs
        .iter()
        .map(|pack| pack.pack_path.clone())
        .collect::<Vec<_>>();

    Ok(BuildResponse {
        pack_path: pack_paths.first().cloned().unwrap_or_else(String::new),
        pack_paths,
        manifest_path: manifest_path.to_string_lossy().into_owned(),
        pack_count: packs.len(),
        project_root: descriptor.root.to_string_lossy().into_owned(),
        project_name: descriptor.config.name.clone(),
        project_config: descriptor.config.clone(),
        startup_scene,
        created_unix_ms,
        bytes_written: packs.iter().map(|pack| pack.bytes_written).sum(),
        asset_count: packs.iter().map(|pack| pack.asset_count).sum(),
        chunk_count: packs.iter().map(|pack| pack.chunk_count).sum(),
        deduped_assets: packs.iter().map(|pack| pack.deduped_assets).sum(),
        elapsed_ms: started.elapsed().as_millis(),
        key_fingerprint: hex_encode(&key_fingerprint_bytes),
        warnings,
        packs,
    })
}

fn build_pack_manifest(
    descriptor: &ProjectDescriptor,
    output_path: &Path,
    manifest_path: &Path,
    pack_stats: &[PackWriteStats],
    created_unix_ms: u64,
    key_fingerprint_bytes: &[u8; 16],
) -> PackSetManifest {
    let manifest_root = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    let packs = pack_stats
        .iter()
        .map(|stats| PackSetPack {
            file: relative_or_absolute_path(&manifest_root, &stats.pack_path),
            asset_count: stats.asset_count,
            chunk_count: stats.chunk_count,
            deduped_assets: stats.deduped_assets,
            bytes_written: stats.bytes_written,
        })
        .collect::<Vec<_>>();

    PackSetManifest {
        version: PACK_SET_MANIFEST_VERSION,
        created_unix_ms,
        project_root: descriptor.root.to_string_lossy().into_owned(),
        key_fingerprint: hex_encode(key_fingerprint_bytes),
        output_base: output_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("pack.hpk")
            .to_string(),
        packs,
    }
}

fn write_pack_manifest(path: &Path, manifest: &PackSetManifest) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create manifest directory {}: {}",
                parent.to_string_lossy(),
                err
            )
        })?;
    }

    let payload = serde_json::to_vec_pretty(manifest)
        .map_err(|err| format!("Failed to encode pack manifest JSON: {err}"))?;

    fs::write(path, payload).map_err(|err| {
        format!(
            "Failed to write pack manifest {}: {}",
            path.to_string_lossy(),
            err
        )
    })
}

fn relative_or_absolute_path(base_dir: &Path, path: &Path) -> String {
    if let Ok(rel) = path.strip_prefix(base_dir) {
        normalize_asset_relative_path(rel)
    } else {
        normalize_asset_relative_path(path)
    }
}

fn write_single_pack(
    output_path: &Path,
    assets: &[ProcessedAsset],
    created_unix_ms: u64,
    key_fingerprint_bytes: [u8; 16],
) -> Result<PackWriteStats, String> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create output directory {}: {}",
                parent.to_string_lossy(),
                err
            )
        })?;
    }

    let mut file = fs::File::create(output_path).map_err(|err| {
        format!(
            "Failed to create pack {}: {}",
            output_path.to_string_lossy(),
            err
        )
    })?;

    file.write_all(&[0u8; PACK_HEADER_SIZE])
        .map_err(|err| format!("Failed to reserve pack header: {err}"))?;

    let mut entries = Vec::with_capacity(assets.len());
    let mut path_blob = Vec::new();
    let mut deduped_assets = 0usize;
    let mut chunk_count = 0usize;

    let mut seen_chunks: HashMap<(u64, u64, u32), (u64, u64)> = HashMap::new();

    for asset in assets {
        align_writer(&mut file, CHUNK_ALIGN)?;

        let dedupe_key = (
            asset.chunk_hash,
            asset.chunk_payload.len() as u64,
            asset.chunk_crc32,
        );

        let (chunk_offset, chunk_len, reused) =
            if let Some((offset, len)) = seen_chunks.get(&dedupe_key) {
                (*offset, *len, true)
            } else {
                let offset = file
                    .stream_position()
                    .map_err(|err| format!("Failed to query pack offset: {err}"))?;
                file.write_all(&asset.chunk_payload)
                    .map_err(|err| format!("Failed to write chunk payload: {err}"))?;
                let len = asset.chunk_payload.len() as u64;
                seen_chunks.insert(dedupe_key, (offset, len));
                chunk_count += 1;
                (offset, len, false)
            };

        if reused {
            deduped_assets += 1;
        }

        let path_offset = path_blob.len() as u64;
        path_blob.extend_from_slice(&asset.path_blob);

        entries.push(PackEntry {
            asset_id: asset.asset_id,
            asset_kind: asset.asset_kind,
            source_flags: asset.source_flags,
            chunk_offset,
            chunk_len,
            source_len: asset.source_len,
            compressed_len: asset.compressed_len,
            source_crc32: asset.source_crc32,
            chunk_crc32: asset.chunk_crc32,
            source_hash: asset.source_hash,
            chunk_hash: asset.chunk_hash,
            path_blob_offset: path_offset,
            path_blob_len: asset.path_blob.len() as u32,
            source_mtime_unix_ms: asset.source_mtime_unix_ms,
        });
    }

    align_writer(&mut file, CHUNK_ALIGN)?;
    let path_blob_offset = file
        .stream_position()
        .map_err(|err| format!("Failed to query path blob offset: {err}"))?;
    file.write_all(&path_blob)
        .map_err(|err| format!("Failed to write path blob: {err}"))?;

    let toc = PackToc {
        version: PACK_VERSION,
        created_unix_ms,
        key_fingerprint: key_fingerprint_bytes,
        entries,
    };
    let toc_bytes = bincode::serialize(&toc)
        .map_err(|err| format!("Failed to encode pack table of contents: {err}"))?;

    align_writer(&mut file, CHUNK_ALIGN)?;
    let toc_offset = file
        .stream_position()
        .map_err(|err| format!("Failed to query TOC offset: {err}"))?;
    file.write_all(&toc_bytes)
        .map_err(|err| format!("Failed to write TOC payload: {err}"))?;

    let bytes_written = file
        .stream_position()
        .map_err(|err| format!("Failed to query pack size: {err}"))?;

    let header = PackHeader {
        flags: PACK_FLAG_COMPRESSED
            | PACK_FLAG_TRANSFORMED
            | PACK_FLAG_PATHS_TRANSFORMED
            | PACK_FLAG_CHUNK_DEDUP,
        created_unix_ms,
        key_fingerprint: key_fingerprint_bytes,
        asset_count: toc.entries.len() as u64,
        chunk_count: chunk_count as u64,
        path_blob_offset,
        path_blob_len: path_blob.len() as u64,
        toc_offset,
        toc_len: toc_bytes.len() as u64,
        toc_hash: fnv1a64(&toc_bytes),
    };

    file.rewind()
        .map_err(|err| format!("Failed to rewind pack for header write: {err}"))?;
    file.write_all(&header.encode())
        .map_err(|err| format!("Failed to write pack header: {err}"))?;
    file.flush()
        .map_err(|err| format!("Failed to flush output pack: {err}"))?;

    Ok(PackWriteStats {
        pack_path: output_path.to_path_buf(),
        asset_count: toc.entries.len(),
        chunk_count,
        deduped_assets,
        bytes_written,
    })
}

fn shard_assets(
    assets: &[ProcessedAsset],
    max_pack_bytes: u64,
    max_assets_per_pack: usize,
) -> Vec<Range<usize>> {
    if assets.is_empty() {
        return vec![0..0];
    }

    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut current_bytes = PACK_HEADER_SIZE as u64;
    let mut current_assets = 0usize;

    for (index, asset) in assets.iter().enumerate() {
        let estimate = estimated_asset_bytes(asset);
        let exceeds_bytes =
            current_assets > 0 && current_bytes.saturating_add(estimate) > max_pack_bytes;
        let exceeds_assets = current_assets >= max_assets_per_pack;

        if exceeds_bytes || exceeds_assets {
            ranges.push(start..index);
            start = index;
            current_bytes = PACK_HEADER_SIZE as u64;
            current_assets = 0;
        }

        current_bytes = current_bytes.saturating_add(estimate);
        current_assets += 1;
    }

    ranges.push(start..assets.len());
    ranges
}

fn estimated_asset_bytes(asset: &ProcessedAsset) -> u64 {
    asset.chunk_payload.len() as u64
        + asset.path_blob.len() as u64
        + ASSET_ENTRY_OVERHEAD_ESTIMATE
        + CHUNK_ALIGN
}

fn output_paths_for_shards(base_output: &Path, shard_count: usize) -> Vec<PathBuf> {
    if shard_count <= 1 {
        return vec![base_output.to_path_buf()];
    }

    let parent = base_output.parent().unwrap_or_else(|| Path::new("."));
    let stem = base_output
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("pack");
    let ext = base_output
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("hpk");

    (0..shard_count)
        .map(|index| parent.join(format!("{}_{}.{}", stem, format!("{index:04}"), ext)))
        .collect()
}

fn manifest_path_for_output(base_output: &Path) -> PathBuf {
    let parent = base_output.parent().unwrap_or_else(|| Path::new("."));
    let stem = base_output
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("pack");
    parent.join(format!("{stem}.packs.json"))
}

fn resolve_startup_scene(
    descriptor: &ProjectDescriptor,
    warnings: &mut Vec<String>,
) -> Result<String, String> {
    if let Some(explicit) = descriptor
        .config
        .startup_scene
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if let Some(path) = resolve_startup_scene_path_hint(descriptor, explicit) {
            if !is_supported_startup_scene(&path) {
                return Err(format!(
                    "Configured startup scene '{}' is unsupported. Supported extensions: .hscene.ron, .hscene, .scene, .glb, .gltf",
                    explicit
                ));
            }
            return Ok(relative_or_absolute_path(&descriptor.root, &path));
        }

        return Err(format!(
            "Configured startup scene '{}' does not exist in project root/assets/scenes",
            explicit
        ));
    }

    let mut discovered = Vec::new();
    if descriptor.layout.scenes_root.exists() {
        for entry in WalkDir::new(&descriptor.layout.scenes_root)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path().to_path_buf();
            if is_supported_startup_scene(&path) {
                discovered.push(path);
            }
        }
    }

    discovered.sort_by(|a, b| a.cmp(b));
    if let Some(path) = discovered.into_iter().next() {
        let selected = relative_or_absolute_path(&descriptor.root, &path);
        warnings.push(format!(
            "No startup_scene configured; using discovered scene '{}'",
            selected
        ));
        return Ok(selected);
    }

    Err(
        "No startup scene configured and no scene files found in project scenes directory"
            .to_string(),
    )
}

fn resolve_startup_scene_path_hint(descriptor: &ProjectDescriptor, value: &str) -> Option<PathBuf> {
    let candidate = PathBuf::from(value);
    if candidate.is_absolute() {
        if candidate.is_file() {
            return Some(candidate);
        }
        return None;
    }

    let project_relative = descriptor.root.join(&candidate);
    if project_relative.is_file() {
        return Some(project_relative);
    }

    let asset_relative = descriptor.layout.assets_root.join(&candidate);
    if asset_relative.is_file() {
        return Some(asset_relative);
    }

    let scene_relative = descriptor.layout.scenes_root.join(&candidate);
    if scene_relative.is_file() {
        return Some(scene_relative);
    }

    None
}

fn is_supported_startup_scene(path: &Path) -> bool {
    let path_lc = path.to_string_lossy().to_ascii_lowercase();
    path_lc.ends_with(".hscene.ron")
        || path_lc.ends_with(".hscene")
        || path_lc.ends_with(".scene")
        || path_lc.ends_with(".glb")
        || path_lc.ends_with(".gltf")
}

fn collect_assets(
    descriptor: &ProjectDescriptor,
    include_hidden: bool,
) -> Result<Vec<PendingAsset>, String> {
    let mut assets = Vec::new();
    let root = &descriptor.layout.assets_root;
    if !root.exists() {
        return Ok(assets);
    }

    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if entry.file_type().is_dir() {
            continue;
        }

        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if !include_hidden && file_name.starts_with('.') {
            continue;
        }

        let rel = match path.strip_prefix(root) {
            Ok(rel) => normalize_asset_relative_path(rel),
            Err(_) => continue,
        };

        let source_flags = classify_source_flags(path);
        let asset_kind = classify_asset_kind(path, source_flags);
        let source_mtime_unix_ms = file_mtime_unix_ms(path);

        assets.push(PendingAsset {
            source_path: path.to_path_buf(),
            relative_path: rel,
            asset_kind,
            source_flags,
            source_mtime_unix_ms,
        });
    }

    Ok(assets)
}

fn build_precompiled_rust_script_assets(
    descriptor: &ProjectDescriptor,
    assets: &[PendingAsset],
    output_path: &Path,
    warnings: &mut Vec<String>,
) -> Result<Vec<PendingAsset>, String> {
    let manifests = discover_rust_script_manifests(&descriptor.layout.assets_root, assets);
    if manifests.is_empty() {
        return Ok(Vec::new());
    }

    ensure_project_rust_script_sdk(&descriptor.root)?;

    let stage_root = output_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".helmer_build")
        .join(format!("rust_scripts_{}", now_unix_ms()));
    let stage_assets_root = stage_root.join("assets");
    let stage_target_root = stage_root.join("target");
    fs::create_dir_all(&stage_assets_root).map_err(|err| {
        format!(
            "Failed to create Rust script staging assets root {}: {}",
            stage_assets_root.to_string_lossy(),
            err
        )
    })?;
    fs::create_dir_all(&stage_target_root).map_err(|err| {
        format!(
            "Failed to create Rust script staging target root {}: {}",
            stage_target_root.to_string_lossy(),
            err
        )
    })?;

    let mut generated_assets = Vec::with_capacity(manifests.len());
    for (index, (manifest_path, manifest_relative)) in manifests.into_iter().enumerate() {
        let target_dir = stage_target_root.join(format!(
            "{:04}_{}",
            index,
            sanitize_path_component(&manifest_relative.to_string_lossy())
        ));
        let artifact_path = build_rust_script_artifact(&manifest_path, &target_dir)?;
        let Some(prebuilt_relative) = rust_prebuilt_plugin_relative_path(&manifest_relative) else {
            return Err(format!(
                "Cannot derive prebuilt plugin path for Rust script manifest {}",
                manifest_relative.to_string_lossy()
            ));
        };
        let staged_output = stage_assets_root.join(&prebuilt_relative);
        if let Some(parent) = staged_output.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "Failed to create Rust prebuilt output directory {}: {}",
                    parent.to_string_lossy(),
                    err
                )
            })?;
        }
        fs::copy(&artifact_path, &staged_output).map_err(|err| {
            format!(
                "Failed to copy Rust script plugin {} to {}: {}",
                artifact_path.to_string_lossy(),
                staged_output.to_string_lossy(),
                err
            )
        })?;

        generated_assets.push(PendingAsset {
            source_path: staged_output.clone(),
            relative_path: normalize_asset_relative_path(&prebuilt_relative),
            asset_kind: AssetKind::Binary,
            source_flags: 0,
            source_mtime_unix_ms: file_mtime_unix_ms(&staged_output),
        });
    }

    warnings.push(format!(
        "Precompiled {} Rust script plugin(s) for runtime builds",
        generated_assets.len()
    ));
    Ok(generated_assets)
}

fn discover_rust_script_manifests(
    assets_root: &Path,
    assets: &[PendingAsset],
) -> Vec<(PathBuf, PathBuf)> {
    let mut manifests = Vec::new();
    let mut seen = HashSet::<PathBuf>::new();

    for asset in assets {
        let Some(manifest_path) = resolve_rust_script_manifest(&asset.source_path) else {
            continue;
        };
        let Ok(manifest_relative) = manifest_path.strip_prefix(assets_root) else {
            continue;
        };
        let manifest_relative = manifest_relative.to_path_buf();
        let is_manifest = manifest_relative
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.eq_ignore_ascii_case("cargo.toml"))
            .unwrap_or(false);
        if !is_manifest {
            continue;
        }
        if seen.insert(manifest_relative.clone()) {
            manifests.push((manifest_path, manifest_relative));
        }
    }

    manifests.sort_by(|a, b| a.1.cmp(&b.1));
    manifests
}

fn build_rust_script_artifact(manifest_path: &Path, target_dir: &Path) -> Result<PathBuf, String> {
    let (_, lib_name) = rust_manifest_names(manifest_path)?;
    fs::create_dir_all(target_dir).map_err(|err| {
        format!(
            "Failed to create Rust script target directory {}: {}",
            target_dir.to_string_lossy(),
            err
        )
    })?;

    let output = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--manifest-path")
        .arg(manifest_path)
        .env("CARGO_TARGET_DIR", target_dir)
        .output()
        .map_err(|err| {
            format!(
                "Failed to run cargo build for Rust script {}: {}",
                manifest_path.to_string_lossy(),
                err
            )
        })?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "cargo build failed for Rust script {}\n{}\n{}",
            manifest_path.to_string_lossy(),
            stdout,
            stderr
        ));
    }

    let file_name = format!(
        "{}{}.{}",
        rust_dynamic_library_prefix(),
        lib_name,
        rust_dynamic_library_extension()
    );
    let artifact_path = target_dir.join("release").join(file_name);
    if !artifact_path.is_file() {
        return Err(format!(
            "Rust script build artifact missing for {}: expected {}",
            manifest_path.to_string_lossy(),
            artifact_path.to_string_lossy()
        ));
    }

    Ok(artifact_path)
}

fn ensure_project_rust_script_sdk(project_root: &Path) -> Result<(), String> {
    let sdk_root = project_root.join(".helmer").join("helmer_script_sdk");
    let sdk_src = sdk_root.join("src");
    fs::create_dir_all(&sdk_src).map_err(|err| {
        format!(
            "Failed to create Rust script SDK directory {}: {}",
            sdk_src.to_string_lossy(),
            err
        )
    })?;
    write_file_if_changed(
        &sdk_root.join("Cargo.toml"),
        EMBEDDED_SCRIPT_SDK_CARGO_TOML.as_bytes(),
    )?;
    write_file_if_changed(
        &sdk_src.join("lib.rs"),
        EMBEDDED_SCRIPT_SDK_LIB_RS.as_bytes(),
    )?;
    Ok(())
}

fn write_file_if_changed(path: &Path, contents: &[u8]) -> Result<(), String> {
    let should_write = match fs::read(path) {
        Ok(existing) => existing != contents,
        Err(_) => true,
    };
    if should_write {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "Failed to create parent directory {}: {}",
                    parent.to_string_lossy(),
                    err
                )
            })?;
        }
        fs::write(path, contents)
            .map_err(|err| format!("Failed to write file {}: {}", path.to_string_lossy(), err))?;
    }
    Ok(())
}

fn sanitize_path_component(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            output.push(ch);
        } else {
            output.push('_');
        }
    }
    while output.ends_with('_') {
        output.pop();
    }
    if output.is_empty() {
        "rust_script".to_string()
    } else {
        output
    }
}

fn process_asset(
    asset: &PendingAsset,
    key_material: &[u8],
    compression_level: u32,
) -> Result<ProcessedAsset, String> {
    let source_bytes = fs::read(&asset.source_path).map_err(|err| {
        format!(
            "Failed to read asset {}: {}",
            asset.source_path.to_string_lossy(),
            err
        )
    })?;

    let source_crc32 = crc32(&source_bytes);
    let source_hash = fnv1a64(&source_bytes);

    let compressed = compress_bytes(&source_bytes, compression_level)?;
    let compressed_len = compressed.len() as u64;

    let normalized_key = asset.relative_path.to_ascii_lowercase();
    let asset_id = fnv1a64(normalized_key.as_bytes());

    let mut chunk_payload = compressed;
    transform_in_place(
        &mut chunk_payload,
        key_material,
        payload_stream_id(asset_id, source_hash),
    );

    let chunk_crc32 = crc32(&chunk_payload);
    let chunk_hash = fnv1a64(&chunk_payload);

    let mut path_blob = asset.relative_path.as_bytes().to_vec();
    transform_in_place(&mut path_blob, key_material, path_stream_id(asset_id));

    Ok(ProcessedAsset {
        relative_path: asset.relative_path.clone(),
        asset_id,
        asset_kind: asset.asset_kind,
        source_flags: asset.source_flags,
        source_mtime_unix_ms: asset.source_mtime_unix_ms,
        source_len: source_bytes.len() as u64,
        source_crc32,
        source_hash,
        compressed_len,
        chunk_payload,
        chunk_crc32,
        chunk_hash,
        path_blob,
    })
}

fn compress_bytes(payload: &[u8], compression_level: u32) -> Result<Vec<u8>, String> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(compression_level));
    encoder
        .write_all(payload)
        .map_err(|err| format!("Failed to deflate asset payload: {err}"))?;
    encoder
        .finish()
        .map_err(|err| format!("Failed to finalize deflate stream: {err}"))
}

fn align_writer(file: &mut fs::File, align: u64) -> Result<(), String> {
    let pos = file
        .stream_position()
        .map_err(|err| format!("Failed to query writer offset: {err}"))?;
    let padding = (align - (pos % align)) % align;
    if padding > 0 {
        let zeros = vec![0u8; padding as usize];
        file.write_all(&zeros)
            .map_err(|err| format!("Failed to write alignment padding: {err}"))?;
    }
    Ok(())
}

fn classify_source_flags(path: &Path) -> u32 {
    let mut flags = 0u32;

    if is_script_path(path) {
        flags |= 1 << 0;
    }

    let is_text_like = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "txt"
                    | "json"
                    | "ron"
                    | "toml"
                    | "yaml"
                    | "yml"
                    | "lua"
                    | "luau"
                    | "hvs"
                    | "md"
                    | "wgsl"
                    | "glsl"
                    | "vert"
                    | "frag"
                    | "rs"
            )
        })
        .unwrap_or(false);

    if is_text_like {
        flags |= 1 << 1;
    }

    flags
}

fn classify_asset_kind(path: &Path, source_flags: u32) -> AssetKind {
    if (source_flags & (1 << 0)) != 0 {
        return AssetKind::Script;
    }

    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return AssetKind::Binary;
    };

    match ext.to_ascii_lowercase().as_str() {
        "glb" | "gltf" | "obj" | "fbx" => AssetKind::Model,
        "png" | "jpg" | "jpeg" | "tga" | "bmp" | "dds" | "ktx" | "ktx2" | "hdr" | "exr" | "qoi" => {
            AssetKind::Texture
        }
        "ron" => {
            let path_lc = path.to_string_lossy().to_ascii_lowercase();
            if path_lc.contains("materials") {
                AssetKind::Material
            } else if path_lc.contains("scenes") {
                AssetKind::Scene
            } else {
                AssetKind::Binary
            }
        }
        "scene" | "hscene" => AssetKind::Scene,
        "wav" | "ogg" | "mp3" | "flac" => AssetKind::Audio,
        "anim" | "animation" => AssetKind::Animation,
        "wgsl" | "spv" | "glsl" | "vert" | "frag" => AssetKind::Shader,
        "ttf" | "otf" => AssetKind::Font,
        _ => AssetKind::Binary,
    }
}

fn file_mtime_unix_ms(path: &Path) -> i64 {
    let Ok(meta) = fs::metadata(path) else {
        return -1;
    };
    let Ok(mtime) = meta.modified() else {
        return -1;
    };
    let Ok(duration) = mtime.duration_since(UNIX_EPOCH) else {
        return -1;
    };
    duration.as_millis() as i64
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn crc32(data: &[u8]) -> u32 {
    let mut hasher = Crc32::new();
    hasher.update(data);
    hasher.finalize()
}
