#[cfg(target_arch = "wasm32")]
fn main() {
    helmer::runtime::asset_worker::link_worker();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("helmer_asset_worker is only supported on wasm32.");
}
