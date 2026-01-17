#![allow(dead_code)]

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum ResourceKind {
    Mesh = 1,
    Material = 2,
    Texture = 3,
    TextureView = 4,
    Sampler = 5,
    Buffer = 6,
    External = 7,  // swapchain, depth, sky, IBL, etc.
    Transient = 8, // frame-local intermediates
}

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct ResourceId(pub u64);

// Layout: [ kind:8 | gen:24 | index:32 ]
impl ResourceId {
    #[inline]
    pub fn new(kind: ResourceKind, index: u32, generation: u32) -> Self {
        let kind_bits = (kind as u64) << 56;
        let gen_bits = ((generation as u64) & 0x00FF_FFFF) << 32;
        let idx_bits = index as u64;
        Self(kind_bits | gen_bits | idx_bits)
    }

    /// Construct directly from an asset server id.
    /// Generation is 0 by default - we rely on asset ids being unique/stable.
    #[inline]
    pub fn from_asset(kind: ResourceKind, asset_id: u32) -> Self {
        Self::new(kind, asset_id, 0)
    }

    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }

    #[inline]
    pub fn kind(self) -> ResourceKind {
        // Safety: we only ever construct with valid enum values
        unsafe { std::mem::transmute::<u8, ResourceKind>((self.0 >> 56) as u8) }
    }

    #[inline]
    pub fn index(self) -> usize {
        (self.0 & 0xFFFF_FFFF) as usize
    }

    #[inline]
    pub fn generation(self) -> u32 {
        ((self.0 >> 32) & 0x00FF_FFFF) as u32
    }
}
