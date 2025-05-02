mod ecs;

use helmer_rs::{
    ecs::component::Component,
    renderer::mev_renderer::{RenderableComponent, ShaderType},
    runtime::Runtime,
};
use proc::Component;

#[derive(Debug, Component)]
struct Position(i64, i64);

fn main() {
    let mut runtime = Runtime::new(|app| {
        let entity = app.ecs.write().unwrap().create_entity();
        app.ecs.write().unwrap().add_component(
            entity,
            RenderableComponent {
                visible: true,
                shader_type: ShaderType::Quad,
            },
        );

        println!(
            "entity's components: {:?}",
            app.ecs.read().unwrap().get_components(entity)
        );
        println!(
            "getting entity's position comp: {:?}",
            app.ecs.read().unwrap().get_component::<Position>(entity)
        );

        let entity = app.ecs.write().unwrap().create_entity();
        app.ecs.write().unwrap().add_component(
            entity,
            RenderableComponent {
                visible: true,
                shader_type: ShaderType::Triangle,
            },
        );
    });
    runtime.init();
}
