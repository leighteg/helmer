#[derive(Debug, Clone, Default)]
pub struct RuntimeBootstrapConfig {
    pub(crate) asset_base_path: Option<String>,
    #[cfg(target_arch = "wasm32")]
    pub wasm_harness: Option<helmer_window::wasm_harness::WasmHarnessConfig>,
}

impl RuntimeBootstrapConfig {
    pub fn set_asset_base_path(&mut self, path: impl Into<String>) {
        self.asset_base_path = Some(path.into());
    }

    pub fn clear_asset_base_path(&mut self) {
        self.asset_base_path = None;
    }

    pub fn asset_base_path(&self) -> Option<&str> {
        self.asset_base_path.as_deref()
    }
}

#[cfg(target_arch = "wasm32")]
impl helmer_window::wasm_harness::WasmHarnessTarget for RuntimeBootstrapConfig {
    fn set_wasm_harness_config(&mut self, config: helmer_window::wasm_harness::WasmHarnessConfig) {
        self.wasm_harness = Some(config);
    }
}
