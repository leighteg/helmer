mod ecs;

use helmer_rs::{
    ecs::component::Component,
    provided::world_space::{Position, Scale},
    runtime::Runtime
};

fn main() {
    let mut runtime = Runtime::new(|app| {
        let mut ecs = app.ecs.write().unwrap();

        let entity = ecs.create_entity();
        ecs.add_component(entity, Position(0.0, 0.0, 0.0));
        ecs.add_component(entity, Scale(1.0, 1.0, 1.0));

        println!(
            "entity's components: {:?}",
            ecs.get_components(entity)
        );
        println!(
            "getting entity's position comp: {:?}",
            ecs.get_component::<Position>(entity)
        );

        let pos = ecs.get_component::<Position>(entity);
        pos.unwrap().0;
    });
    runtime.init();
}
