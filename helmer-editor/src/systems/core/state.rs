use helmer_engine::ecs::system::System;

#[derive(Copy, Clone, PartialEq)]
pub enum WorldState {
    Edit,
    Play,
}

pub struct EditorStateResource {
    pub world_state: WorldState,
}

pub struct EditorStateSystem {
    pub last_world_state: WorldState,
}

impl System for EditorStateSystem {
    fn name(&self) -> &str {
        "editor state system"
    }

    fn run(
        &mut self,
        dt: f32,
        ecs: &mut helmer_engine::ecs::ecs_core::ECSCore,
        input_manager: &helmer_engine::runtime::input_manager::InputManager,
    ) {
        ecs.resource_scope::<EditorStateResource, _>(|ecs, editor_state_resource| {
            if editor_state_resource.world_state != self.last_world_state {}
            self.last_world_state = editor_state_resource.world_state;
        });
    }
}
