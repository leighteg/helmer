use crate::egui_integration::EguiResource;

pub struct InspectorUI {}

impl InspectorUI {
    pub fn add_window(egui_res: &mut EguiResource) {
        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc| {}),
            "inspector".to_string(),
        ));
    }
}
