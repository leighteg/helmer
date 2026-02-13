use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub const PATH_STREAM_SALT: u64 = 0xA5A5_5A5A_F0F0_0F0F;

pub fn parse_key_material(raw: &str) -> Result<Vec<u8>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("Build key is required and cannot be empty".to_string());
    }

    let key_bytes = if let Some(hex) = trimmed.strip_prefix("hex:") {
        decode_hex(hex)?
    } else {
        trimmed.as_bytes().to_vec()
    };

    if key_bytes.len() < 8 {
        return Err("Build key must be at least 8 bytes".to_string());
    }

    Ok(key_bytes)
}

pub fn key_fingerprint(key: &[u8]) -> [u8; 16] {
    let mut reversed = key.to_vec();
    reversed.reverse();

    let h0 = fnv1a64(key).to_le_bytes();
    let h1 = fnv1a64(&reversed).to_le_bytes();

    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&h0);
    out[8..].copy_from_slice(&h1);
    out
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

pub fn payload_stream_id(asset_id: u64, source_hash: u64) -> u64 {
    asset_id ^ source_hash
}

pub fn path_stream_id(asset_id: u64) -> u64 {
    asset_id ^ PATH_STREAM_SALT
}

pub fn transform_in_place(bytes: &mut [u8], key: &[u8], stream_id: u64) {
    if bytes.is_empty() {
        return;
    }

    let seed = derive_seed(key, stream_id);
    let mut rng = ChaCha20Rng::from_seed(seed);

    let mut offset = 0usize;
    let mut keystream = [0u8; 4096];
    while offset < bytes.len() {
        let remaining = bytes.len() - offset;
        let chunk_len = remaining.min(keystream.len());
        rng.fill_bytes(&mut keystream[..chunk_len]);
        for (dst, src) in bytes[offset..offset + chunk_len]
            .iter_mut()
            .zip(&keystream[..chunk_len])
        {
            *dst ^= *src;
        }
        offset += chunk_len;
    }
}

pub fn fnv1a64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn decode_hex(raw: &str) -> Result<Vec<u8>, String> {
    let bytes = raw.as_bytes();
    if bytes.len() % 2 != 0 {
        return Err("hex key must have an even number of characters".to_string());
    }

    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut idx = 0usize;
    while idx < bytes.len() {
        let hi = hex_nibble(bytes[idx]).ok_or_else(|| {
            format!(
                "Invalid hex character '{}' in build key",
                bytes[idx] as char
            )
        })?;
        let lo = hex_nibble(bytes[idx + 1]).ok_or_else(|| {
            format!(
                "Invalid hex character '{}' in build key",
                bytes[idx + 1] as char
            )
        })?;
        out.push((hi << 4) | lo);
        idx += 2;
    }

    Ok(out)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(10 + (byte - b'a')),
        b'A'..=b'F' => Some(10 + (byte - b'A')),
        _ => None,
    }
}

fn derive_seed(key: &[u8], stream_id: u64) -> [u8; 32] {
    let mut seed = [0u8; 32];
    let stream_bytes = stream_id.to_le_bytes();
    for index in 0..32 {
        let key_byte = key[index % key.len()];
        let stream_byte = stream_bytes[index % stream_bytes.len()];
        seed[index] = key_byte.wrapping_add((index as u8).rotate_left((index % 7) as u32))
            ^ stream_byte
            ^ ((index as u8).wrapping_mul(31));
    }
    seed
}
