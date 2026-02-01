use wasm_bindgen::JsCast;
use web_sys::{Document, Element, HtmlCanvasElement};

use crate::runtime::runtime::Runtime;

#[derive(Debug, Clone)]
pub struct WasmHarnessConfig {
    pub canvas_id: Option<String>,
    pub mount_id: Option<String>,
    pub asset_base_path: Option<String>,
    pub opfs_enabled: bool,
    pub prevent_default: bool,
    pub focusable: bool,
    pub append_canvas: bool,
    pub fit_canvas_to_parent: bool,
}

impl Default for WasmHarnessConfig {
    fn default() -> Self {
        Self {
            canvas_id: Some("helmer-canvas".to_string()),
            mount_id: None,
            asset_base_path: None,
            opfs_enabled: true,
            prevent_default: true,
            focusable: true,
            append_canvas: true,
            fit_canvas_to_parent: true,
        }
    }
}

impl WasmHarnessConfig {
    pub fn apply_to_runtime<T: 'static>(&self, runtime: &mut Runtime<T>) -> Result<(), String> {
        let canvas = resolve_canvas(self)?;
        runtime.configure_web_window(Some(canvas), false, self.prevent_default, self.focusable);

        runtime.set_opfs_enabled(self.opfs_enabled);
        if let Some(base_path) = &self.asset_base_path {
            runtime.set_asset_base_path(base_path.clone());
        }

        Ok(())
    }
}

fn resolve_canvas(config: &WasmHarnessConfig) -> Result<HtmlCanvasElement, String> {
    let window = web_sys::window().ok_or_else(|| "missing window".to_string())?;
    let document = window
        .document()
        .ok_or_else(|| "missing document".to_string())?;

    let canvas = if let Some(id) = &config.canvas_id {
        match document.get_element_by_id(id) {
            Some(element) => element
                .dyn_into::<HtmlCanvasElement>()
                .map_err(|_| format!("element '{}' is not a canvas", id))?,
            None => {
                let canvas = create_canvas(&document)?;
                canvas.set_id(id);
                canvas
            }
        }
    } else {
        create_canvas(&document)?
    };

    if config.fit_canvas_to_parent {
        let style = canvas.style();
        style
            .set_property("width", "100%")
            .map_err(js_err_to_string)?;
        style
            .set_property("height", "100%")
            .map_err(js_err_to_string)?;
        style
            .set_property("display", "block")
            .map_err(js_err_to_string)?;
        style
            .set_property("outline", "none")
            .map_err(js_err_to_string)?;
    }

    if config.append_canvas {
        let parent: Element = if let Some(mount_id) = &config.mount_id {
            document
                .get_element_by_id(mount_id)
                .ok_or_else(|| format!("mount element '{}' not found", mount_id))?
        } else {
            document
                .body()
                .ok_or_else(|| "missing document body".to_string())?
                .into()
        };

        if canvas.parent_node().is_none() {
            parent.append_child(&canvas).map_err(js_err_to_string)?;
        }
    }

    if config.fit_canvas_to_parent {
        sync_canvas_resolution(&canvas)?;
    }

    Ok(canvas)
}

fn create_canvas(document: &Document) -> Result<HtmlCanvasElement, String> {
    document
        .create_element("canvas")
        .map_err(js_err_to_string)?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| "failed to create canvas element".to_string())
}

fn js_err_to_string(err: impl std::fmt::Debug) -> String {
    format!("{:?}", err)
}

fn sync_canvas_resolution(canvas: &HtmlCanvasElement) -> Result<(), String> {
    let window = web_sys::window().ok_or_else(|| "missing window".to_string())?;
    let dpr = window.device_pixel_ratio();
    let client_width = canvas.client_width() as f64;
    let client_height = canvas.client_height() as f64;
    let mut width = (client_width * dpr).round() as u32;
    let mut height = (client_height * dpr).round() as u32;

    if width == 0 || height == 0 {
        let fallback_width = window
            .inner_width()
            .ok()
            .and_then(|value| value.as_f64())
            .unwrap_or(1.0);
        let fallback_height = window
            .inner_height()
            .ok()
            .and_then(|value| value.as_f64())
            .unwrap_or(1.0);
        width = (fallback_width * dpr).round() as u32;
        height = (fallback_height * dpr).round() as u32;
    }

    canvas.set_width(width.max(1));
    canvas.set_height(height.max(1));
    Ok(())
}
