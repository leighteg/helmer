#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use web_sys::{Document, Element, HtmlCanvasElement};

#[derive(Debug, Clone)]
pub struct WasmHarnessConfig {
    pub canvas_id: Option<String>,
    pub mount_id: Option<String>,
    pub append_canvas: bool,
    pub fit_canvas_to_parent: bool,
}

impl Default for WasmHarnessConfig {
    fn default() -> Self {
        Self {
            canvas_id: Some("helmer-canvas".to_string()),
            mount_id: None,
            append_canvas: true,
            fit_canvas_to_parent: true,
        }
    }
}

pub trait WasmHarnessTarget {
    fn set_wasm_harness_config(&mut self, config: WasmHarnessConfig);
}

impl WasmHarnessConfig {
    pub fn apply_to_runtime<T>(&self, runtime: &mut T) -> Result<(), String>
    where
        T: WasmHarnessTarget,
    {
        runtime.set_wasm_harness_config(self.clone());
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
impl WasmHarnessConfig {
    pub fn ensure_canvas(&self) -> Result<HtmlCanvasElement, String> {
        let window = web_sys::window().ok_or_else(|| "missing window".to_string())?;
        let document = window
            .document()
            .ok_or_else(|| "missing document".to_string())?;

        let canvas = if let Some(id) = &self.canvas_id {
            match document.get_element_by_id(id) {
                Some(existing) => existing
                    .dyn_into::<HtmlCanvasElement>()
                    .map_err(|_| format!("element '{id}' is not a canvas"))?,
                None => {
                    let canvas = create_canvas(&document)?;
                    canvas.set_id(id);
                    canvas
                }
            }
        } else {
            create_canvas(&document)?
        };

        if self.fit_canvas_to_parent {
            let style = canvas.style();
            style
                .set_property("width", "100%")
                .map_err(|err| format!("{err:?}"))?;
            style
                .set_property("height", "100%")
                .map_err(|err| format!("{err:?}"))?;
            style
                .set_property("display", "block")
                .map_err(|err| format!("{err:?}"))?;
        }

        if self.append_canvas && canvas.parent_node().is_none() {
            let parent: Element = if let Some(mount_id) = &self.mount_id {
                document
                    .get_element_by_id(mount_id)
                    .ok_or_else(|| format!("mount element '{mount_id}' not found"))?
            } else {
                document
                    .body()
                    .ok_or_else(|| "missing document body".to_string())?
                    .into()
            };
            parent
                .append_child(&canvas)
                .map_err(|err| format!("{err:?}"))?;
        }

        Ok(canvas)
    }
}

#[cfg(target_arch = "wasm32")]
fn create_canvas(document: &Document) -> Result<HtmlCanvasElement, String> {
    use wasm_bindgen::JsCast;

    document
        .create_element("canvas")
        .map_err(|err| format!("{err:?}"))?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| "failed to create canvas element".to_string())
}
