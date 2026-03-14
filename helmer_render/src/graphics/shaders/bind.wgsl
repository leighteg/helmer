// helmer_bind.wgsl
// Unified Helmer bindless interface (Option D: fully unified).
// Backends:
// - Modern: texture_binding_array & storage binding arrays
// - Fallback: fixed-size arrays + indirection tables

#ifndef HEL_BINDLESS
    #error "HEL_BINDLESS must be defined by backend."
#endif

// Backend-specific defines:
// HEL_BINDLESS_MODERN
// HEL_BINDLESS_FALLBACK

// -------- Texture & sampler arrays --------

struct HelTextureHandle {
    index: u32,
};

struct HelSamplerHandle {
    index: u32;
};

struct HelBufferHandle {
    index: u32;
    word_offset: u32;
};

#if defined(HEL_BINDLESS_MODERN)

// Modern: direct arrays
@group(0) @binding(0)
var hel_textures : array<texture_2d<f32>>;

@group(0) @binding(1)
var hel_samplers : array<sampler>;

@group(0) @binding(2)
var<storage, read> hel_buffers : array<u32>;

fn hel_sample_texture(htex: HelTextureHandle, hsam: HelSamplerHandle, uv: vec2<f32>) -> vec4<f32> {
    return textureSample(hel_textures[htex.index], hel_samplers[hsam.index], uv);
}

fn hel_read_u32(hbuf: HelBufferHandle) -> u32 {
    let idx = hbuf.index + hbuf.word_offset;
    return hel_buffers[idx];
}

#else // HEL_BINDLESS_FALLBACK

// Fallback: same interface, but arrays may be smaller and actually accessed via indirection
// tables that you can inject here later if needed.

@group(0) @binding(0)
var hel_textures : array<texture_2d<f32>>;

@group(0) @binding(1)
var hel_samplers : array<sampler>;

@group(0) @binding(2)
var<storage, read> hel_buffers : array<u32>;

fn hel_sample_texture(htex: HelTextureHandle, hsam: HelSamplerHandle, uv: vec2<f32>) -> vec4<f32> {
    return textureSample(hel_textures[htex.index], hel_samplers[hsam.index], uv);
}

fn hel_read_u32(hbuf: HelBufferHandle) -> u32 {
    let idx = hbuf.index + hbuf.word_offset;
    return hel_buffers[idx];
}

#endif