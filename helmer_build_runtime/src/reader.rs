use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use crc32fast::Hasher as Crc32;
use flate2::read::ZlibDecoder;

use crate::{
    crypto::{
        fnv1a64, key_fingerprint, parse_key_material, path_stream_id, payload_stream_id,
        transform_in_place,
    },
    format::{PACK_HEADER_SIZE, PackEntry, PackHeader, PackSetManifest, PackToc},
};

#[derive(Debug, Clone)]
struct LoadedPack {
    path: PathBuf,
    bytes: Vec<u8>,
    header: PackHeader,
    toc: PackToc,
}

#[derive(Debug, Clone)]
struct AssetLocation {
    pack_index: usize,
    entry: PackEntry,
}

#[derive(Debug)]
pub struct PackSetReader {
    manifest: PackSetManifest,
    key_material: Vec<u8>,
    packs: Vec<LoadedPack>,
    index: HashMap<String, AssetLocation>,
}

impl PackSetReader {
    pub fn open(manifest_path: impl AsRef<Path>, key: &str) -> Result<Self, String> {
        let manifest_path = manifest_path.as_ref().to_path_buf();
        let manifest_dir = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        let key_material = parse_key_material(key)?;
        let expected_key_fingerprint = key_fingerprint(&key_material);

        let manifest = read_manifest(&manifest_path)?;

        if manifest.key_fingerprint.to_ascii_lowercase()
            != crate::crypto::hex_encode(&expected_key_fingerprint)
        {
            return Err("Build key does not match pack-set manifest fingerprint".to_string());
        }

        let mut packs = Vec::with_capacity(manifest.packs.len());
        for pack in &manifest.packs {
            let pack_path = resolve_manifest_pack_path(&manifest_dir, &pack.file);
            let bytes = fs::read(&pack_path).map_err(|err| {
                format!(
                    "Failed to read pack {}: {}",
                    pack_path.to_string_lossy(),
                    err
                )
            })?;

            if bytes.len() < PACK_HEADER_SIZE {
                return Err(format!(
                    "Pack {} is too small to contain a valid header",
                    pack_path.to_string_lossy()
                ));
            }

            let header = PackHeader::decode(&bytes[..PACK_HEADER_SIZE])?;
            if header.key_fingerprint != expected_key_fingerprint {
                return Err(format!(
                    "Pack key fingerprint mismatch in {}",
                    pack_path.to_string_lossy()
                ));
            }

            let toc_start = usize::try_from(header.toc_offset).map_err(|_| {
                format!("TOC offset out of range in {}", pack_path.to_string_lossy())
            })?;
            let toc_len = usize::try_from(header.toc_len).map_err(|_| {
                format!("TOC length out of range in {}", pack_path.to_string_lossy())
            })?;
            let toc_end = toc_start
                .checked_add(toc_len)
                .ok_or_else(|| format!("TOC range overflow in {}", pack_path.to_string_lossy()))?;

            if toc_end > bytes.len() {
                return Err(format!(
                    "TOC range is outside file bounds in {}",
                    pack_path.to_string_lossy()
                ));
            }

            let toc_bytes = &bytes[toc_start..toc_end];
            if fnv1a64(toc_bytes) != header.toc_hash {
                return Err(format!(
                    "TOC hash mismatch in {}",
                    pack_path.to_string_lossy()
                ));
            }

            let toc: PackToc = bincode::deserialize(toc_bytes).map_err(|err| {
                format!(
                    "Failed to decode TOC for {}: {err}",
                    pack_path.to_string_lossy()
                )
            })?;

            packs.push(LoadedPack {
                path: pack_path,
                bytes,
                header,
                toc,
            });
        }

        let mut index = HashMap::new();
        for (pack_index, loaded) in packs.iter().enumerate() {
            let path_blob_base = usize::try_from(loaded.header.path_blob_offset).map_err(|_| {
                format!(
                    "Path blob offset out of range in {}",
                    loaded.path.to_string_lossy()
                )
            })?;

            for entry in &loaded.toc.entries {
                let entry_path_offset = usize::try_from(entry.path_blob_offset).map_err(|_| {
                    format!(
                        "Path blob entry offset out of range in {}",
                        loaded.path.to_string_lossy()
                    )
                })?;
                let path_start =
                    path_blob_base
                        .checked_add(entry_path_offset)
                        .ok_or_else(|| {
                            format!(
                                "Path blob start overflow in {}",
                                loaded.path.to_string_lossy()
                            )
                        })?;
                let path_end = path_start
                    .checked_add(entry.path_blob_len as usize)
                    .ok_or_else(|| {
                        format!(
                            "Path blob end overflow in {}",
                            loaded.path.to_string_lossy()
                        )
                    })?;

                if path_end > loaded.bytes.len() {
                    return Err(format!(
                        "Path blob range outside file bounds in {}",
                        loaded.path.to_string_lossy()
                    ));
                }

                let mut path_bytes = loaded.bytes[path_start..path_end].to_vec();
                transform_in_place(
                    &mut path_bytes,
                    &key_material,
                    path_stream_id(entry.asset_id),
                );

                let relative_path = String::from_utf8(path_bytes).map_err(|_| {
                    format!(
                        "Decoded path blob is not UTF-8 in {}",
                        loaded.path.to_string_lossy()
                    )
                })?;

                if index.contains_key(&relative_path) {
                    return Err(format!(
                        "Duplicate asset path '{}' found in pack set",
                        relative_path
                    ));
                }

                index.insert(
                    relative_path,
                    AssetLocation {
                        pack_index,
                        entry: entry.clone(),
                    },
                );
            }
        }

        Ok(Self {
            manifest,
            key_material,
            packs,
            index,
        })
    }

    pub fn manifest(&self) -> &PackSetManifest {
        &self.manifest
    }

    pub fn list_assets(&self) -> Vec<String> {
        let mut assets = self.index.keys().cloned().collect::<Vec<_>>();
        assets.sort();
        assets
    }

    pub fn contains_asset(&self, relative_path: &str) -> bool {
        self.index.contains_key(relative_path)
    }

    pub fn read_asset(&self, relative_path: &str) -> Result<Vec<u8>, String> {
        let Some(location) = self.index.get(relative_path) else {
            return Err(format!(
                "Asset '{}' was not found in pack set",
                relative_path
            ));
        };

        let loaded = &self.packs[location.pack_index];
        let entry = &location.entry;

        let chunk_start = usize::try_from(entry.chunk_offset).map_err(|_| {
            format!(
                "Chunk offset out of range for '{}' in {}",
                relative_path,
                loaded.path.to_string_lossy()
            )
        })?;
        let chunk_len = usize::try_from(entry.chunk_len).map_err(|_| {
            format!(
                "Chunk length out of range for '{}' in {}",
                relative_path,
                loaded.path.to_string_lossy()
            )
        })?;
        let chunk_end = chunk_start.checked_add(chunk_len).ok_or_else(|| {
            format!(
                "Chunk range overflow for '{}' in {}",
                relative_path,
                loaded.path.to_string_lossy()
            )
        })?;

        if chunk_end > loaded.bytes.len() {
            return Err(format!(
                "Chunk range outside file bounds for '{}' in {}",
                relative_path,
                loaded.path.to_string_lossy()
            ));
        }

        let chunk_encrypted = &loaded.bytes[chunk_start..chunk_end];
        if crc32(chunk_encrypted) != entry.chunk_crc32 {
            return Err(format!(
                "Chunk CRC mismatch for '{}' in {}",
                relative_path,
                loaded.path.to_string_lossy()
            ));
        }

        let mut chunk_compressed = chunk_encrypted.to_vec();
        transform_in_place(
            &mut chunk_compressed,
            &self.key_material,
            payload_stream_id(entry.asset_id, entry.source_hash),
        );

        let mut source = Vec::with_capacity(entry.source_len as usize);
        let mut decoder = ZlibDecoder::new(chunk_compressed.as_slice());
        decoder.read_to_end(&mut source).map_err(|err| {
            format!(
                "Failed to inflate '{}' from {}: {}",
                relative_path,
                loaded.path.to_string_lossy(),
                err
            )
        })?;

        if source.len() as u64 != entry.source_len {
            return Err(format!(
                "Source length mismatch for '{}': expected {}, got {}",
                relative_path,
                entry.source_len,
                source.len()
            ));
        }

        if crc32(&source) != entry.source_crc32 {
            return Err(format!(
                "Source CRC mismatch for '{}' in {}",
                relative_path,
                loaded.path.to_string_lossy()
            ));
        }

        Ok(source)
    }
}

fn read_manifest(path: &Path) -> Result<PackSetManifest, String> {
    let raw = fs::read_to_string(path).map_err(|err| {
        format!(
            "Failed to read pack manifest {}: {}",
            path.to_string_lossy(),
            err
        )
    })?;
    serde_json::from_str(&raw).map_err(|err| {
        format!(
            "Failed to parse pack manifest {}: {}",
            path.to_string_lossy(),
            err
        )
    })
}

fn resolve_manifest_pack_path(manifest_dir: &Path, pack_file: &str) -> PathBuf {
    let pack = PathBuf::from(pack_file);
    if pack.is_absolute() {
        pack
    } else {
        manifest_dir.join(pack)
    }
}

fn crc32(data: &[u8]) -> u32 {
    let mut hasher = Crc32::new();
    hasher.update(data);
    hasher.finalize()
}
