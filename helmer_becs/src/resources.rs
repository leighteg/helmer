use crossbeam_channel::Sender;
use hashbrown::HashMap;
use std::{
    collections::VecDeque,
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::Arc,
};

use bevy_ecs::prelude::ReflectResource;
use bevy_ecs::resource::Resource;
use bevy_reflect::Reflect;
use helmer_asset::runtime::asset_server::AssetServer;
#[cfg(not(target_arch = "wasm32"))]
use helmer_audio::AudioHostId;
use helmer_audio::{AudioBackend, AudioBus, AudioOutputDevice, AudioOutputSettings};
use helmer_render::{
    graphics::common::renderer::{RenderMessage, RendererStats, StreamingTuning},
    runtime::{RuntimeConfig, RuntimeProfiling, RuntimeTuning},
};
use helmer_window::runtime::{
    input_manager::InputManager,
    runtime::{PerformanceMetrics, RuntimeCursorState, RuntimeWindowControl},
};
use parking_lot::{Mutex, RwLock};

use crate::profiling::SystemProfiler;

#[derive(Clone)]
#[cfg_attr(not(target_arch = "wasm32"), derive(Resource))]
pub struct BecsAssetServer(pub Arc<Mutex<AssetServer>>);

#[cfg(target_arch = "wasm32")]
pub type BecsAssetServerParam<'w> = bevy_ecs::system::NonSend<'w, BecsAssetServer>;
#[cfg(not(target_arch = "wasm32"))]
pub type BecsAssetServerParam<'w> = bevy_ecs::prelude::Res<'w, BecsAssetServer>;

#[derive(Resource)]
pub struct BecsInputManager(pub Arc<RwLock<InputManager>>);
#[derive(Resource)]
pub struct BecsPerformanceMetrics(pub Arc<PerformanceMetrics>);
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct BecsRuntimeConfig(pub RuntimeConfig);
#[derive(Resource)]
pub struct BecsRuntimeTuning(pub Arc<RuntimeTuning>);
#[derive(Resource)]
pub struct BecsRuntimeProfiling(pub Arc<RuntimeProfiling>);
#[derive(Resource)]
pub struct BecsSystemProfiler(pub Arc<SystemProfiler>);
#[derive(Resource)]
pub struct BecsRuntimeCursorState(pub Arc<RuntimeCursorState>);
#[derive(Resource)]
pub struct BecsRuntimeWindowControl(pub Arc<RuntimeWindowControl>);
#[derive(Resource)]
pub struct BecsRendererStats(pub Arc<RendererStats>);
#[derive(Resource)]
pub struct BecsRenderSender(pub Sender<RenderMessage>);
#[derive(Resource, Clone, Copy, Debug)]
pub struct BecsStreamingTuning(pub StreamingTuning);
#[derive(Resource, Clone, Copy, Debug)]
pub struct BecsLodTuning(pub crate::systems::render_system::LodTuning);
#[derive(Resource, Clone, Copy, Debug)]
pub struct BecsRenderWorkerTuning(pub crate::systems::render_system::RenderWorkerTuning);
#[derive(Resource, Clone, Copy, Debug)]
pub struct BecsSceneTuning(pub crate::systems::scene_system::SceneTuning);

#[derive(Resource, Default, Clone)]
pub struct DebugGraphHistory {
    pub vram_bytes: VecDeque<f64>,
    pub mesh_bytes: VecDeque<f64>,
    pub texture_bytes: VecDeque<f64>,
    pub material_bytes: VecDeque<f64>,
    pub audio_bytes: VecDeque<f64>,
    pub fps: VecDeque<f64>,
    pub tps: VecDeque<f64>,
}

#[derive(Resource, Clone)]
pub struct AudioBackendResource(pub Arc<AudioBackend>);

#[derive(Resource, Default, Clone)]
pub struct ProfilingHistory {
    pub main_event_ms: VecDeque<f64>,
    pub main_update_ms: VecDeque<f64>,
    pub logic_frame_ms: VecDeque<f64>,
    pub logic_asset_ms: VecDeque<f64>,
    pub logic_input_ms: VecDeque<f64>,
    pub logic_tick_ms: VecDeque<f64>,
    pub logic_schedule_ms: VecDeque<f64>,
    pub logic_render_send_ms: VecDeque<f64>,
    pub ecs_render_data_ms: VecDeque<f64>,
    pub ecs_scene_spawn_ms: VecDeque<f64>,
    pub ecs_scene_update_ms: VecDeque<f64>,
    pub render_thread_frame_ms: VecDeque<f64>,
    pub render_thread_messages_ms: VecDeque<f64>,
    pub render_thread_upload_ms: VecDeque<f64>,
    pub render_thread_render_ms: VecDeque<f64>,
    pub render_prepare_globals_ms: VecDeque<f64>,
    pub render_streaming_plan_ms: VecDeque<f64>,
    pub render_occlusion_ms: VecDeque<f64>,
    pub render_graph_ms: VecDeque<f64>,
    pub render_graph_pass_ms: VecDeque<f64>,
    pub render_graph_encoder_create_ms: VecDeque<f64>,
    pub render_graph_encoder_finish_ms: VecDeque<f64>,
    pub render_graph_overhead_ms: VecDeque<f64>,
    pub render_resource_mgmt_ms: VecDeque<f64>,
    pub render_acquire_ms: VecDeque<f64>,
    pub render_submit_ms: VecDeque<f64>,
    pub render_present_ms: VecDeque<f64>,
    pub ui_system_ms: VecDeque<f64>,
    pub ui_run_frame_ms: VecDeque<f64>,
    pub ui_interaction_ms: VecDeque<f64>,
    pub ui_scroll_metrics_ms: VecDeque<f64>,
    pub ui_render_data_convert_ms: VecDeque<f64>,
    pub render_ui_build_ms: VecDeque<f64>,
    pub render_pass_ms: HashMap<String, VecDeque<f64>>,
    pub render_pass_last_ms: HashMap<String, f64>,
    pub render_pass_order: Vec<String>,
    pub audio_mix_ms: VecDeque<f64>,
    pub audio_callback_ms: VecDeque<f64>,
    pub audio_emitters: VecDeque<f64>,
    pub audio_streaming_emitters: VecDeque<f64>,
}

#[derive(Resource, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Resource)]
pub struct DeltaTime(pub f32);

#[derive(Resource, Clone, Debug, Default)]
pub struct DraggedFile(pub Option<PathBuf>);

// Unprefixed aliases for BECS app-facing resource wrappers.
pub type AssetServerResource = BecsAssetServer;
pub type AssetServerParam<'w> = BecsAssetServerParam<'w>;
pub type InputManagerResource = BecsInputManager;
pub type PerformanceMetricsResource = BecsPerformanceMetrics;
pub type RuntimeConfigResource = BecsRuntimeConfig;
pub type RuntimeTuningResource = BecsRuntimeTuning;
pub type RuntimeProfilingResource = BecsRuntimeProfiling;
pub type SystemProfilerResource = BecsSystemProfiler;
pub type RuntimeCursorStateResource = BecsRuntimeCursorState;
pub type RuntimeWindowControlResource = BecsRuntimeWindowControl;
pub type RendererStatsResource = BecsRendererStats;
pub type RenderSenderResource = BecsRenderSender;
pub type StreamingTuningResource = BecsStreamingTuning;
pub type LodTuningResource = BecsLodTuning;
pub type RenderWorkerTuningResource = BecsRenderWorkerTuning;
pub type SceneTuningResource = BecsSceneTuning;

impl BecsAssetServer {
    pub fn cloned(&self) -> Self {
        Self(self.0.clone())
    }

    pub fn lock(&self) -> parking_lot::MutexGuard<'_, AssetServer> {
        self.0.lock()
    }
}

impl Deref for BecsAssetServer {
    type Target = Arc<Mutex<AssetServer>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl BecsInputManager {
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, InputManager> {
        self.0.read()
    }

    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, InputManager> {
        self.0.write()
    }
}

impl Deref for BecsRuntimeConfig {
    type Target = RuntimeConfig;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BecsRuntimeConfig {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl BecsSystemProfiler {
    #[inline]
    pub fn enabled(&self) -> bool {
        self.0.enabled()
    }

    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.0.set_enabled(enabled);
    }

    #[inline]
    pub fn reset_all(&self) {
        self.0.reset_all();
    }

    #[inline]
    pub fn snapshots(&self) -> Vec<crate::profiling::SystemProfileSnapshot> {
        self.0.snapshots()
    }

    #[inline]
    pub fn auto_enable_new_systems(&self) -> bool {
        self.0.auto_enable_new_systems()
    }

    #[inline]
    pub fn set_auto_enable_new_systems(&self, enabled: bool) {
        self.0.set_auto_enable_new_systems(enabled);
    }

    #[inline]
    pub fn set_all_systems_enabled(&self, enabled: bool) {
        self.0.set_all_systems_enabled(enabled);
    }

    #[inline]
    pub fn set_system_enabled(&self, name: &str, enabled: bool) -> bool {
        self.0.set_system_enabled(name, enabled)
    }

    #[inline]
    pub fn begin_scope(&self, name: &'static str) -> Option<crate::profiling::SystemProfileScope> {
        self.0.begin_scope(name)
    }
}

impl Deref for BecsSystemProfiler {
    type Target = SystemProfiler;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl BecsRenderSender {
    #[inline]
    pub fn send(
        &self,
        msg: RenderMessage,
    ) -> Result<(), crossbeam_channel::SendError<RenderMessage>> {
        self.0.send(msg)
    }
}

impl Deref for BecsRenderSender {
    type Target = Sender<RenderMessage>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AudioBackendResource {
    #[inline]
    pub fn enabled(&self) -> bool {
        self.0.enabled()
    }

    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.0.set_enabled(enabled);
    }

    #[inline]
    pub fn clear_emitters(&self) {
        self.0.clear_emitters();
    }

    #[inline]
    pub fn output_settings(&self) -> AudioOutputSettings {
        self.0.output_settings()
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    pub fn available_output_hosts(&self) -> Vec<AudioHostId> {
        self.0.available_output_hosts()
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    pub fn available_output_devices(&self, host_id: Option<AudioHostId>) -> Vec<AudioOutputDevice> {
        self.0.available_output_devices(host_id)
    }

    #[inline]
    pub fn reconfigure(&self, settings: AudioOutputSettings) -> Result<(), String> {
        self.0.reconfigure(settings)
    }

    #[inline]
    pub fn last_error(&self) -> Option<String> {
        self.0.last_error()
    }

    #[inline]
    pub fn bus_list(&self) -> Vec<AudioBus> {
        self.0.bus_list()
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    pub fn bus_name(&self, bus: AudioBus) -> String {
        self.0.bus_name(bus)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    pub fn set_bus_name(&self, bus: AudioBus, name: String) {
        self.0.set_bus_name(bus, name);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    pub fn create_custom_bus(&self, name: Option<String>) -> AudioBus {
        self.0.create_custom_bus(name)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    pub fn remove_bus(&self, bus: AudioBus) {
        self.0.remove_bus(bus);
    }

    #[inline]
    pub fn bus_volume(&self, bus: AudioBus) -> f32 {
        self.0.bus_volume(bus)
    }

    #[inline]
    pub fn set_bus_volume(&self, bus: AudioBus, volume: f32) {
        self.0.set_bus_volume(bus, volume);
    }

    #[inline]
    pub fn scene_volume(&self, scene_id: u64) -> f32 {
        self.0.scene_volume(scene_id)
    }

    #[inline]
    pub fn set_scene_volume(&self, scene_id: u64, volume: f32) {
        self.0.set_scene_volume(scene_id, volume);
    }

    #[inline]
    pub fn head_width(&self) -> f32 {
        self.0.head_width()
    }

    #[inline]
    pub fn set_head_width(&self, width: f32) {
        self.0.set_head_width(width);
    }

    #[inline]
    pub fn speed_of_sound(&self) -> f32 {
        self.0.speed_of_sound()
    }

    #[inline]
    pub fn set_speed_of_sound(&self, speed: f32) {
        self.0.set_speed_of_sound(speed);
    }

    #[inline]
    pub fn streaming_config(&self) -> (usize, usize) {
        self.0.streaming_config()
    }

    #[inline]
    pub fn set_streaming_config(&self, buffer_frames: usize, chunk_frames: usize) {
        self.0.set_streaming_config(buffer_frames, chunk_frames);
    }
}

impl Deref for AudioBackendResource {
    type Target = AudioBackend;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Default for BecsStreamingTuning {
    fn default() -> Self {
        Self(StreamingTuning::default())
    }
}

impl Default for BecsLodTuning {
    fn default() -> Self {
        Self(crate::systems::render_system::LodTuning::default())
    }
}

impl Default for BecsRenderWorkerTuning {
    fn default() -> Self {
        Self(crate::systems::render_system::RenderWorkerTuning::default())
    }
}

impl Default for BecsSceneTuning {
    fn default() -> Self {
        Self(crate::systems::scene_system::SceneTuning::default())
    }
}
