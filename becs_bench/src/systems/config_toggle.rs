use bevy_ecs::system::{Res, ResMut};
use helmer_becs::{BevyInputManager, BevyRuntimeConfig};
use winit::keyboard::KeyCode;

pub fn config_toggle_system(
    input_manager: Res<BevyInputManager>,
    mut runtime_config: ResMut<BevyRuntimeConfig>,
) {
    let input_manager = input_manager.0.read();
    let runtime_config = &mut runtime_config.0;

    if input_manager.is_key_active(KeyCode::ControlLeft) {
        return;
    }

    for key in input_manager.just_pressed.iter() {
        match key {
            KeyCode::KeyZ => {
                runtime_config.render_config.shadow_pass = !runtime_config.render_config.shadow_pass
            }
            KeyCode::KeyG => {
                runtime_config.render_config.ssgi_pass = !runtime_config.render_config.ssgi_pass
            }
            KeyCode::KeyH => {
                runtime_config.render_config.sky_pass = !runtime_config.render_config.sky_pass
            }
            KeyCode::KeyR => {
                runtime_config.render_config.ssr_pass = !runtime_config.render_config.ssr_pass
            }
            KeyCode::KeyF => {
                runtime_config.render_config.frustum_culling =
                    !runtime_config.render_config.frustum_culling
            }
            KeyCode::KeyL => runtime_config.render_config.lod = !runtime_config.render_config.lod,
            KeyCode::KeyU => runtime_config.egui = !runtime_config.egui,

            KeyCode::Digit0 => runtime_config.render_config.shader_constants.shade_mode = 1,
            KeyCode::Digit1 => runtime_config.render_config.shader_constants.shade_mode = 0,
            KeyCode::Digit2 => runtime_config.render_config.shader_constants.shade_mode = 2,

            _ => {}
        }
    }
}
