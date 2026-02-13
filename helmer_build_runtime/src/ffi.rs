use std::{panic, ptr, slice};

use serde::{Deserialize, Serialize};

use crate::builder::{BuildRequest, BuildResponse, build_project_pack};

pub const HELMER_BUILD_ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HelmerBuildBuffer {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
}

impl HelmerBuildBuffer {
    fn empty() -> Self {
        Self {
            ptr: ptr::null_mut(),
            len: 0,
            cap: 0,
        }
    }

    fn from_vec(mut vec: Vec<u8>) -> Self {
        let buffer = Self {
            ptr: vec.as_mut_ptr(),
            len: vec.len(),
            cap: vec.capacity(),
        };
        std::mem::forget(vec);
        buffer
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct BuildResultEnvelope {
    ok: bool,
    response: Option<BuildResponse>,
    error: Option<String>,
}

#[unsafe(no_mangle)]
pub extern "C" fn helmer_build_abi_version() -> u32 {
    HELMER_BUILD_ABI_VERSION
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn helmer_build_pack_json(
    request_ptr: *const u8,
    request_len: usize,
    out_buffer: *mut HelmerBuildBuffer,
) -> i32 {
    if out_buffer.is_null() {
        return 10;
    }

    // SAFETY: caller provided an out pointer; we initialize it immediately.
    unsafe {
        *out_buffer = HelmerBuildBuffer::empty();
    }

    let result = panic::catch_unwind(|| {
        if request_ptr.is_null() || request_len == 0 {
            return Err("Request payload is empty".to_string());
        }

        // SAFETY: non-null pointer and explicit length are validated above.
        let payload = unsafe { slice::from_raw_parts(request_ptr, request_len) };
        let request: BuildRequest = serde_json::from_slice(payload)
            .map_err(|err| format!("Failed to decode build request JSON: {err}"))?;

        build_project_pack(&request)
    });

    let envelope = match result {
        Ok(Ok(response)) => BuildResultEnvelope {
            ok: true,
            response: Some(response),
            error: None,
        },
        Ok(Err(err)) => BuildResultEnvelope {
            ok: false,
            response: None,
            error: Some(err),
        },
        Err(_) => BuildResultEnvelope {
            ok: false,
            response: None,
            error: Some("Build runtime panicked while processing request".to_string()),
        },
    };

    let encoded = match serde_json::to_vec(&envelope) {
        Ok(data) => data,
        Err(err) => {
            let fallback = BuildResultEnvelope {
                ok: false,
                response: None,
                error: Some(format!("Failed to encode build response JSON: {err}")),
            };
            serde_json::to_vec(&fallback).unwrap_or_default()
        }
    };

    let buffer = HelmerBuildBuffer::from_vec(encoded);

    // SAFETY: out_buffer was checked for null at function entry.
    unsafe {
        *out_buffer = buffer;
    }

    if envelope.ok { 0 } else { 1 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn helmer_build_free_buffer(buffer: HelmerBuildBuffer) {
    if buffer.ptr.is_null() || buffer.cap == 0 {
        return;
    }

    // SAFETY: buffer originates from HelmerBuildBuffer::from_vec in this library.
    unsafe {
        let _ = Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.cap);
    }
}
