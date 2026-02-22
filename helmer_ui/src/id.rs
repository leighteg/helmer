use std::fmt;

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UiId(pub u64);

impl UiId {
    pub const ROOT: Self = Self(0xcbf2_9ce4_8422_2325);

    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub fn from_str(value: &str) -> Self {
        Self(fnv1a_64(value.as_bytes()))
    }

    pub fn child<K: IntoUiId>(self, key: K) -> Self {
        let key = key.into_ui_id();
        Self(hash_pair(self.0, key.0))
    }
}

impl fmt::Debug for UiId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UiId({:#018x})", self.0)
    }
}

impl fmt::Display for UiId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#018x}", self.0)
    }
}

pub trait IntoUiId {
    fn into_ui_id(self) -> UiId;
}

impl IntoUiId for UiId {
    fn into_ui_id(self) -> UiId {
        self
    }
}

impl IntoUiId for u64 {
    fn into_ui_id(self) -> UiId {
        UiId(self)
    }
}

impl IntoUiId for usize {
    fn into_ui_id(self) -> UiId {
        UiId(self as u64)
    }
}

impl IntoUiId for &str {
    fn into_ui_id(self) -> UiId {
        UiId::from_str(self)
    }
}

impl IntoUiId for String {
    fn into_ui_id(self) -> UiId {
        UiId::from_str(&self)
    }
}

impl IntoUiId for &String {
    fn into_ui_id(self) -> UiId {
        UiId::from_str(self)
    }
}

fn hash_pair(a: u64, b: u64) -> u64 {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&a.to_le_bytes());
    bytes[8..].copy_from_slice(&b.to_le_bytes());
    fnv1a_64(&bytes)
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x1000_0000_01b3;

    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
