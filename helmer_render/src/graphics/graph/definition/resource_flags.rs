#![allow(dead_code)]

bitflags::bitflags! {
    #[derive(Debug, Default, Clone, Copy)]
    pub struct ResourceFlags: u32 {
        /// Never evict unless explicitly forced.
        const PINNED         = 1 << 0;
        /// Prefer to keep in VRAM (bias against eviction).
        const PREFER_RESIDENT = 1 << 1;
        /// OK to drop and re-stream at any time.
        const STREAMING      = 1 << 2;
        /// Transient per-frame temp (should be in transient heap).
        const TRANSIENT      = 1 << 3;
        /// Frequently updated, avoid thrashing.
        const FREQUENT_UPDATE = 1 << 4;
        /// Keep slot reserved so ResourceId stays valid even if evicted.
        const STABLE_ID      = 1 << 5;
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ResourceUsageHints {
    pub flags: ResourceFlags,
    /// Approximate byte size. Used for VRAM budgeting and eviction heuristics.
    pub estimated_size_bytes: u64,
}
