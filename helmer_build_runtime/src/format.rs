use serde::{Deserialize, Serialize};

pub const PACK_MAGIC: [u8; 8] = *b"HLPACK1\0";
pub const PACK_VERSION: u32 = 1;
pub const PACK_HEADER_SIZE: usize = 128;
pub const PACK_SET_MANIFEST_VERSION: u32 = 1;

pub const PACK_FLAG_COMPRESSED: u32 = 1 << 0;
pub const PACK_FLAG_TRANSFORMED: u32 = 1 << 1;
pub const PACK_FLAG_PATHS_TRANSFORMED: u32 = 1 << 2;
pub const PACK_FLAG_CHUNK_DEDUP: u32 = 1 << 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AssetKind {
    Unknown = 0,
    Model = 1,
    Texture = 2,
    Material = 3,
    Scene = 4,
    Audio = 5,
    Script = 6,
    Animation = 7,
    Font = 8,
    Shader = 9,
    Binary = 10,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackToc {
    pub version: u32,
    pub created_unix_ms: u64,
    pub key_fingerprint: [u8; 16],
    pub entries: Vec<PackEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackEntry {
    pub asset_id: u64,
    pub asset_kind: AssetKind,
    pub source_flags: u32,
    pub chunk_offset: u64,
    pub chunk_len: u64,
    pub source_len: u64,
    pub compressed_len: u64,
    pub source_crc32: u32,
    pub chunk_crc32: u32,
    pub source_hash: u64,
    pub chunk_hash: u64,
    pub path_blob_offset: u64,
    pub path_blob_len: u32,
    pub source_mtime_unix_ms: i64,
}

#[derive(Debug, Clone)]
pub struct PackHeader {
    pub flags: u32,
    pub created_unix_ms: u64,
    pub key_fingerprint: [u8; 16],
    pub asset_count: u64,
    pub chunk_count: u64,
    pub path_blob_offset: u64,
    pub path_blob_len: u64,
    pub toc_offset: u64,
    pub toc_len: u64,
    pub toc_hash: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackSetManifest {
    pub version: u32,
    pub created_unix_ms: u64,
    pub project_root: String,
    pub key_fingerprint: String,
    pub output_base: String,
    pub packs: Vec<PackSetPack>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackSetPack {
    pub file: String,
    pub asset_count: usize,
    pub chunk_count: usize,
    pub deduped_assets: usize,
    pub bytes_written: u64,
}

impl PackHeader {
    pub fn encode(&self) -> [u8; PACK_HEADER_SIZE] {
        let mut out = [0u8; PACK_HEADER_SIZE];
        let mut cursor = 0usize;

        out[cursor..cursor + 8].copy_from_slice(&PACK_MAGIC);
        cursor += 8;

        out[cursor..cursor + 4].copy_from_slice(&PACK_VERSION.to_le_bytes());
        cursor += 4;

        out[cursor..cursor + 4].copy_from_slice(&self.flags.to_le_bytes());
        cursor += 4;

        out[cursor..cursor + 8].copy_from_slice(&self.created_unix_ms.to_le_bytes());
        cursor += 8;

        out[cursor..cursor + 16].copy_from_slice(&self.key_fingerprint);
        cursor += 16;

        out[cursor..cursor + 8].copy_from_slice(&self.asset_count.to_le_bytes());
        cursor += 8;

        out[cursor..cursor + 8].copy_from_slice(&self.chunk_count.to_le_bytes());
        cursor += 8;

        out[cursor..cursor + 8].copy_from_slice(&self.path_blob_offset.to_le_bytes());
        cursor += 8;

        out[cursor..cursor + 8].copy_from_slice(&self.path_blob_len.to_le_bytes());
        cursor += 8;

        out[cursor..cursor + 8].copy_from_slice(&self.toc_offset.to_le_bytes());
        cursor += 8;

        out[cursor..cursor + 8].copy_from_slice(&self.toc_len.to_le_bytes());
        cursor += 8;

        out[cursor..cursor + 8].copy_from_slice(&self.toc_hash.to_le_bytes());

        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < PACK_HEADER_SIZE {
            return Err(format!(
                "Pack header too small: expected at least {}, got {}",
                PACK_HEADER_SIZE,
                bytes.len()
            ));
        }

        let mut cursor = 0usize;

        let magic: [u8; 8] = bytes[cursor..cursor + 8]
            .try_into()
            .map_err(|_| "Failed to decode pack magic".to_string())?;
        cursor += 8;
        if magic != PACK_MAGIC {
            return Err("Invalid pack magic".to_string());
        }

        let version = u32::from_le_bytes(
            bytes[cursor..cursor + 4]
                .try_into()
                .map_err(|_| "Failed to decode pack version".to_string())?,
        );
        cursor += 4;
        if version != PACK_VERSION {
            return Err(format!(
                "Unsupported pack version: expected {}, found {}",
                PACK_VERSION, version
            ));
        }

        let flags = u32::from_le_bytes(
            bytes[cursor..cursor + 4]
                .try_into()
                .map_err(|_| "Failed to decode pack flags".to_string())?,
        );
        cursor += 4;

        let created_unix_ms = u64::from_le_bytes(
            bytes[cursor..cursor + 8]
                .try_into()
                .map_err(|_| "Failed to decode pack created timestamp".to_string())?,
        );
        cursor += 8;

        let key_fingerprint: [u8; 16] = bytes[cursor..cursor + 16]
            .try_into()
            .map_err(|_| "Failed to decode pack key fingerprint".to_string())?;
        cursor += 16;

        let asset_count = u64::from_le_bytes(
            bytes[cursor..cursor + 8]
                .try_into()
                .map_err(|_| "Failed to decode pack asset count".to_string())?,
        );
        cursor += 8;

        let chunk_count = u64::from_le_bytes(
            bytes[cursor..cursor + 8]
                .try_into()
                .map_err(|_| "Failed to decode pack chunk count".to_string())?,
        );
        cursor += 8;

        let path_blob_offset = u64::from_le_bytes(
            bytes[cursor..cursor + 8]
                .try_into()
                .map_err(|_| "Failed to decode pack path blob offset".to_string())?,
        );
        cursor += 8;

        let path_blob_len = u64::from_le_bytes(
            bytes[cursor..cursor + 8]
                .try_into()
                .map_err(|_| "Failed to decode pack path blob length".to_string())?,
        );
        cursor += 8;

        let toc_offset = u64::from_le_bytes(
            bytes[cursor..cursor + 8]
                .try_into()
                .map_err(|_| "Failed to decode pack TOC offset".to_string())?,
        );
        cursor += 8;

        let toc_len = u64::from_le_bytes(
            bytes[cursor..cursor + 8]
                .try_into()
                .map_err(|_| "Failed to decode pack TOC length".to_string())?,
        );
        cursor += 8;

        let toc_hash = u64::from_le_bytes(
            bytes[cursor..cursor + 8]
                .try_into()
                .map_err(|_| "Failed to decode pack TOC hash".to_string())?,
        );

        Ok(Self {
            flags,
            created_unix_ms,
            key_fingerprint,
            asset_count,
            chunk_count,
            path_blob_offset,
            path_blob_len,
            toc_offset,
            toc_len,
            toc_hash,
        })
    }
}
