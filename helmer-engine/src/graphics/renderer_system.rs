use std::time::Instant;

use hashbrown::HashMap;
use proc::System;
use crate::{ecs::{ecs_core::{ECSCore, Entity}, system::System}, graphics::renderer::renderer::{RenderData, RenderLight, RenderObject}, provided::components::{ActiveCamera, Camera, Light, MeshRenderer, Transform}, runtime::input_manager::InputManager};

/// An intermediate struct used only within the extraction system
/// to hold a snapshot of the world's state for one frame.
#[derive(Clone, Default)]
pub struct ExtractedState {
    pub objects: HashMap<Entity, (Transform, usize, usize, bool)>, // Transform, mesh_id, material_id, casts_shadow
    pub lights: HashMap<Entity, (Transform, Light)>,
    pub camera_transform: Transform,
    pub camera_component: Camera,
}

/// The resource that will live in the ECS and hold the render data.
/// The Option allows the logic thread to `.take()` the data without blocking.
#[derive(Default)]
pub struct RenderPacket(pub Option<RenderData>);

/// This system is responsible for collecting all data required for rendering
/// from the ECS and packaging it into a `RenderPacket` resource.
/// It should run LAST in the system schedule.
pub struct RenderDataSystem {
    // Stores the state of the last frame to enable motion interpolation.
    previous_state: Option<ExtractedState>,
}

impl RenderDataSystem {
    pub fn new() -> Self {
        Self {
            previous_state: None,
        }
    }
}

impl Default for RenderDataSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl System for RenderDataSystem {
    fn name(&self) -> &str {
        "RenderDataSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        // --- STEP 1: EXTRACT CURRENT STATE FROM THE ECS ---
        // This is a snapshot of all the data we need for this exact frame.
        let mut current_state = ExtractedState::default();

        // Query for all renderable objects
        ecs.component_pool.query_for_each::<(Transform, MeshRenderer), _>(|entity, (transform, mesh_renderer)| {
            if mesh_renderer.visible {
                current_state.objects.insert(entity, (*transform, mesh_renderer.mesh_id, mesh_renderer.material_id, mesh_renderer.casts_shadow));
            }
        });

        // Query for all lights
        ecs.component_pool.query_for_each::<(Transform, Light), _>(|entity, (transform, light)| {
            current_state.lights.insert(entity, (*transform, *light));
        });

        // Find the active camera
        if let Some((cam, trans, _)) = ecs.component_pool.query_exact::<(Camera, Transform, ActiveCamera)>().next() {
            current_state.camera_component = *cam;
            current_state.camera_transform = *trans;
        } else {
            // If no active camera, use default. Your game might want to handle this differently.
            println!("[WARN] No active camera found in scene for rendering.");
        }


        // --- STEP 2: BUILD THE RENDERDATA PACKET ---
        // Use the state from the previous frame for interpolation. If it doesn't exist (i.e., first frame),
        // we use the current state for both previous and current to avoid a pop-in.
        let prev_state = self.previous_state.as_ref().unwrap_or(&current_state);

        let objects = current_state.objects.iter()
            .map(|(&entity, &(current_transform, mesh_id, material_id, casts_shadow))| {
                // Find the previous transform for this entity. Default to the current one if it's a new entity.
                let previous_transform = prev_state.objects.get(&entity).map_or(current_transform, |(prev_trans, _, _, _)| *prev_trans);
                RenderObject {
                    id: entity,
                    mesh_id,
                    material_id,
                    current_transform,
                    previous_transform,
                    casts_shadow,
                }
            })
            .collect();
            
        let lights = current_state.lights.iter()
            .map(|(&entity, &(current_transform, light_component))| {
                let previous_transform = prev_state.lights.get(&entity).map_or(current_transform, |(prev_trans, _)| *prev_trans);
                RenderLight {
                    color: light_component.color.into(),
                    intensity: light_component.intensity,
                    current_transform,
                    previous_transform,
                    light_type: light_component.light_type,
                }
            })
            .collect();
        
        let render_data = RenderData {
            objects,
            lights,
            camera_component: current_state.camera_component,
            current_camera_transform: current_state.camera_transform,
            previous_camera_transform: prev_state.camera_transform,
            timestamp: Instant::now(),
        };

        // --- STEP 3: STORE THE PACKET IN THE RESOURCE POOL ---
        // Get mutable access to the resource. We expect it to be there.
        if let Some(render_packet) = ecs.get_resource_mut::<RenderPacket>() {
            // Place the completed data packet into the resource.
            // The logic thread will `.take()` it after the scheduler finishes.
            render_packet.0 = Some(render_data);
        } else {
            eprintln!("[ERROR] RenderPacket resource not found in ECS. Please register it on startup.");
        }


        // --- STEP 4: UPDATE STATE FOR THE NEXT FRAME ---
        // The current state now becomes the previous state for the next tick.
        self.previous_state = Some(current_state);
    }
}