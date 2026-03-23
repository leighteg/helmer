#[cfg(not(target_arch = "wasm32"))]
use crate::graphics::common::renderer::NativeRenderInit;
use crate::graphics::common::renderer::{
    AssetStreamingRequest, RenderControl, RenderMessage, RendererStats,
    render_message_payload_bytes,
};
use crate::runtime::{RuntimeConfig as RenderRuntimeConfig, RuntimeProfiling, RuntimeTuning};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded, unbounded};
use helmer::runtime::{
    PerformanceMetrics, RuntimeContext, RuntimeError, RuntimeExtension,
    RuntimePerformanceMetricsResource,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::warn;
use web_time::Instant;
use winit::{dpi::PhysicalSize, window::Window};

#[cfg(target_arch = "wasm32")]
use crate::graphics::common::renderer::WgpuBackend;
use crate::graphics::common::renderer::initialize_renderer;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

#[derive(Clone)]
pub struct RenderMessageSender(pub Sender<RenderMessage>);

#[derive(Clone)]
pub struct RenderAssetMessageSender(pub Sender<RenderMessage>);

#[derive(Clone)]
pub struct RenderControlMessageSender(pub Sender<RenderMessage>);

#[derive(Clone)]
pub struct RenderStreamRequestReceiver(pub Receiver<AssetStreamingRequest>);

#[derive(Clone)]
pub struct RenderStatsResource(pub Arc<RendererStats>);

#[derive(Clone)]
pub struct RenderRuntimeTuningResource(pub Arc<RuntimeTuning>);

#[derive(Clone)]
pub struct RenderRuntimeProfilingResource(pub Arc<RuntimeProfiling>);

pub struct RenderExtension {
    render_sender: Sender<RenderMessage>,
    render_receiver: Receiver<RenderMessage>,
    control_sender: Sender<RenderMessage>,
    control_receiver: Receiver<RenderMessage>,
    asset_sender: Sender<RenderMessage>,
    asset_receiver: Receiver<RenderMessage>,
    stream_request_sender: Sender<AssetStreamingRequest>,
    stream_request_receiver: Receiver<AssetStreamingRequest>,
    stats: Arc<RendererStats>,
    metrics: Option<Arc<PerformanceMetrics>>,
    tuning: Arc<RuntimeTuning>,
    profiling: Arc<RuntimeProfiling>,
    running: Arc<AtomicBool>,
    config: RenderRuntimeConfig,
    #[cfg(not(target_arch = "wasm32"))]
    run_on_main_thread: bool,
    #[cfg(not(target_arch = "wasm32"))]
    native_render_state: Option<NativeInlineRenderState>,
    #[cfg(not(target_arch = "wasm32"))]
    native_pending_bootstrap: Option<NativeRendererBootstrap>,
    #[cfg(not(target_arch = "wasm32"))]
    native_pending_render_messages: VecDeque<RenderMessage>,
    #[cfg(not(target_arch = "wasm32"))]
    native_pending_asset_messages: VecDeque<RenderMessage>,
    #[cfg(target_arch = "wasm32")]
    render_state: Option<WebRenderState>,
    #[cfg(target_arch = "wasm32")]
    render_init_receiver: Option<Receiver<RenderInitResult>>,
    #[cfg(target_arch = "wasm32")]
    pending_bootstrap: Option<(Arc<Window>, PhysicalSize<u32>)>,
    #[cfg(target_arch = "wasm32")]
    pending_render_messages: VecDeque<RenderMessage>,
    #[cfg(target_arch = "wasm32")]
    pending_asset_messages: VecDeque<RenderMessage>,
    #[cfg(target_arch = "wasm32")]
    next_renderer_init_retry_at: Option<Instant>,
    #[cfg(target_arch = "wasm32")]
    renderer_init_failures: u32,
}

impl Default for RenderExtension {
    fn default() -> Self {
        let tuning = Arc::new(RuntimeTuning::default());
        let capacity = tuning
            .render_message_capacity
            .load(Ordering::Relaxed)
            .max(1);
        let (render_sender, render_receiver) = bounded(capacity);
        let (control_sender, control_receiver) = unbounded();
        let asset_capacity = tuning.asset_message_capacity.load(Ordering::Relaxed).max(1);
        let (asset_sender, asset_receiver) = bounded(asset_capacity);
        let stream_capacity = tuning
            .asset_stream_queue_capacity
            .load(Ordering::Relaxed)
            .max(1);
        let (stream_sender, stream_receiver) = bounded(stream_capacity);
        Self {
            render_sender,
            render_receiver,
            control_sender,
            control_receiver,
            asset_sender,
            asset_receiver,
            stream_request_sender: stream_sender,
            stream_request_receiver: stream_receiver,
            stats: Arc::new(RendererStats::default()),
            metrics: None,
            tuning,
            profiling: Arc::new(RuntimeProfiling::default()),
            running: Arc::new(AtomicBool::new(false)),
            config: RenderRuntimeConfig::default(),
            #[cfg(not(target_arch = "wasm32"))]
            run_on_main_thread: false,
            #[cfg(not(target_arch = "wasm32"))]
            native_render_state: None,
            #[cfg(not(target_arch = "wasm32"))]
            native_pending_bootstrap: None,
            #[cfg(not(target_arch = "wasm32"))]
            native_pending_render_messages: VecDeque::new(),
            #[cfg(not(target_arch = "wasm32"))]
            native_pending_asset_messages: VecDeque::new(),
            #[cfg(target_arch = "wasm32")]
            render_state: None,
            #[cfg(target_arch = "wasm32")]
            render_init_receiver: None,
            #[cfg(target_arch = "wasm32")]
            pending_bootstrap: None,
            #[cfg(target_arch = "wasm32")]
            pending_render_messages: VecDeque::new(),
            #[cfg(target_arch = "wasm32")]
            pending_asset_messages: VecDeque::new(),
            #[cfg(target_arch = "wasm32")]
            next_renderer_init_retry_at: None,
            #[cfg(target_arch = "wasm32")]
            renderer_init_failures: 0,
        }
    }
}

impl RenderExtension {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: RenderRuntimeConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }
}

impl RuntimeExtension for RenderExtension {
    fn name(&self) -> &'static str {
        "helmer_render"
    }

    fn on_register(&mut self, ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        ctx.resources()
            .insert(RenderMessageSender(self.render_sender.clone()));
        ctx.resources()
            .insert(RenderControlMessageSender(self.control_sender.clone()));
        ctx.resources()
            .insert(RenderAssetMessageSender(self.asset_sender.clone()));
        ctx.resources().insert(RenderStreamRequestReceiver(
            self.stream_request_receiver.clone(),
        ));
        ctx.resources()
            .insert(RenderStatsResource(Arc::clone(&self.stats)));
        ctx.resources()
            .insert(RenderRuntimeTuningResource(Arc::clone(&self.tuning)));
        ctx.resources()
            .insert(RenderRuntimeProfilingResource(Arc::clone(&self.profiling)));
        Ok(())
    }

    fn on_start(&mut self, _ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        if self.running.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        while self.control_receiver.try_recv().is_ok() {}
        self.metrics = _ctx
            .resources()
            .get::<RuntimePerformanceMetricsResource>()
            .map(|resource| resource.0.clone());

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.run_on_main_thread = _ctx.is_single_threaded();
            if self.run_on_main_thread {
                self.native_render_state = None;
                self.native_pending_bootstrap = None;
                self.native_pending_render_messages.clear();
                self.native_pending_asset_messages.clear();
            } else {
                self.spawn_render_thread(_ctx)?;
            }
        }

        Ok(())
    }

    fn on_tick(&mut self, _ctx: &RuntimeContext, _dt: Duration) -> Result<(), RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            self.pump_channels_on_main_thread();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.run_on_main_thread {
                self.pump_channels_on_main_thread_native();
            }
        }

        Ok(())
    }

    fn on_stop(&mut self, _ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        self.running.store(false, Ordering::Release);
        let _ = self.control_sender.send(RenderMessage::Shutdown);
        let _ = self.asset_sender.try_send(RenderMessage::Shutdown);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.release_queued_messages_native();
            self.native_render_state = None;
            self.native_pending_bootstrap = None;
            self.run_on_main_thread = false;
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.release_queued_messages();
            self.render_state = None;
            self.render_init_receiver = None;
            self.pending_bootstrap = None;
            self.next_renderer_init_retry_at = None;
            self.renderer_init_failures = 0;
        }
        Ok(())
    }
}

fn publish_render_fps(
    metrics: Option<&Arc<PerformanceMetrics>>,
    fps_window_start: &mut Instant,
    fps_frames: &mut u32,
    now: Instant,
) {
    let Some(metrics) = metrics else {
        return;
    };
    *fps_frames = fps_frames.saturating_add(1);
    let elapsed = now.saturating_duration_since(*fps_window_start);
    if elapsed >= Duration::from_secs(1) {
        let seconds = elapsed.as_secs_f64().max(f64::EPSILON);
        let fps = ((*fps_frames as f64) / seconds)
            .round()
            .clamp(0.0, u32::MAX as f64) as u32;
        metrics.set_fps(fps);
        *fps_frames = 0;
        *fps_window_start = now;
    }
}

#[cfg(not(target_arch = "wasm32"))]
enum NativeRendererBootstrap {
    Window {
        window: Arc<Window>,
        size: PhysicalSize<u32>,
    },
    Init {
        window: Arc<Window>,
        render_init: NativeRenderInit,
        size: PhysicalSize<u32>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PendingAssetUploadKey {
    Mesh(usize),
    Texture(usize),
    Material(usize),
}

struct PendingAssetUpload {
    sequence: u64,
    message: RenderMessage,
    bytes: usize,
}

#[derive(Default)]
struct PendingAssetUploadQueue {
    next_sequence: u64,
    order: VecDeque<(u64, PendingAssetUploadKey)>,
    entries: HashMap<PendingAssetUploadKey, PendingAssetUpload>,
    total_bytes: usize,
}

impl PendingAssetUploadQueue {
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    fn push(&mut self, message: RenderMessage, bytes: usize) {
        let Some(key) = pending_asset_upload_key(&message) else {
            return;
        };
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        let replaced = self.entries.insert(
            key,
            PendingAssetUpload {
                sequence,
                message,
                bytes,
            },
        );
        if let Some(existing) = replaced {
            self.total_bytes = self.total_bytes.saturating_sub(existing.bytes);
        }
        self.total_bytes = self.total_bytes.saturating_add(bytes);
        self.order.push_back((sequence, key));
    }

    fn pop_front(&mut self) -> Option<(RenderMessage, usize)> {
        while let Some((sequence, key)) = self.order.pop_front() {
            let is_current = self
                .entries
                .get(&key)
                .is_some_and(|entry| entry.sequence == sequence);
            if !is_current {
                continue;
            }
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
                return Some((entry.message, entry.bytes));
            }
        }
        None
    }

    fn remove(&mut self, key: PendingAssetUploadKey) -> Option<usize> {
        let removed = self.entries.remove(&key)?;
        self.total_bytes = self.total_bytes.saturating_sub(removed.bytes);
        Some(removed.bytes)
    }

    fn drain_release(&mut self, tuning: &Arc<RuntimeTuning>) {
        for entry in self.entries.drain().map(|(_, entry)| entry) {
            tuning.release_asset_upload(entry.bytes);
        }
        self.order.clear();
        self.total_bytes = 0;
    }
}

fn pending_asset_upload_key(message: &RenderMessage) -> Option<PendingAssetUploadKey> {
    match message {
        RenderMessage::CreateMesh { id, .. } | RenderMessage::RemoveMesh { id } => {
            Some(PendingAssetUploadKey::Mesh(*id))
        }
        RenderMessage::CreateTexture { id, .. } => Some(PendingAssetUploadKey::Texture(*id)),
        RenderMessage::CreateMaterial(mat) => Some(PendingAssetUploadKey::Material(mat.id)),
        _ => None,
    }
}

fn enqueue_asset_upload(
    asset_backlog: &mut PendingAssetUploadQueue,
    immediate_backlog: &mut PendingAssetUploadQueue,
    message: RenderMessage,
    backlog_enabled: bool,
    max_pending: usize,
    max_pending_bytes: usize,
) -> usize {
    let Some(key) = pending_asset_upload_key(&message) else {
        return 0;
    };
    let bytes = render_message_payload_bytes(&message);
    let released_bytes = asset_backlog
        .remove(key)
        .unwrap_or(0)
        .saturating_add(immediate_backlog.remove(key).unwrap_or(0));
    let fits_bytes = asset_backlog.total_bytes().saturating_add(bytes) <= max_pending_bytes;
    if backlog_enabled && asset_backlog.len() < max_pending && fits_bytes {
        asset_backlog.push(message, bytes);
    } else {
        immediate_backlog.push(message, bytes);
    }
    released_bytes
}

fn asset_message_collection_budget(tuning: &Arc<RuntimeTuning>) -> usize {
    const MIN_ASSET_MESSAGES_PER_FRAME: usize = 32;
    const MAX_ASSET_MESSAGES_PER_FRAME: usize = 256;

    tuning
        .asset_uploads_per_frame
        .load(Ordering::Relaxed)
        .max(1)
        .saturating_mul(32)
        .clamp(MIN_ASSET_MESSAGES_PER_FRAME, MAX_ASSET_MESSAGES_PER_FRAME)
}

fn asset_work_budgets_for_frame(
    base_messages: usize,
    base_uploads: usize,
    prioritize_render: bool,
) -> (usize, usize) {
    if !prioritize_render {
        return (base_messages, base_uploads.max(1));
    }

    (base_messages.min(32), base_uploads.max(1).min(2))
}

fn render_due_for_frame(
    frame_start: Instant,
    last_render: Instant,
    tuning: &Arc<RuntimeTuning>,
) -> bool {
    match tuning.load_target_fps() {
        Some(target_fps) if target_fps.is_finite() && target_fps > 0.0 => {
            let frame_duration = Duration::from_secs_f32(1.0 / target_fps);
            frame_start.duration_since(last_render) >= frame_duration
        }
        _ => true,
    }
}

fn drain_control_messages(
    control_receiver: &Receiver<RenderMessage>,
    pending_render_messages: &mut VecDeque<RenderMessage>,
) -> bool {
    loop {
        match control_receiver.try_recv() {
            Ok(message) => pending_render_messages.push_back(message),
            Err(TryRecvError::Empty) => return true,
            Err(TryRecvError::Disconnected) => return false,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeInlineRenderState {
    renderer: Option<crate::graphics::renderer::GraphRenderer>,
    window: Option<Arc<Window>>,
    surface_size: PhysicalSize<u32>,
    last_render: Instant,
    fps_window_start: Instant,
    fps_frames: u32,
    poll_frame: u32,
    asset_backlog: PendingAssetUploadQueue,
    immediate_backlog: PendingAssetUploadQueue,
    upload_batch: Vec<RenderMessage>,
    upload_bytes: Vec<usize>,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeInlineRenderState {
    fn new(
        renderer: crate::graphics::renderer::GraphRenderer,
        window: Option<Arc<Window>>,
        surface_size: PhysicalSize<u32>,
    ) -> Self {
        let now = Instant::now();
        Self {
            renderer: Some(renderer),
            window,
            surface_size,
            last_render: now,
            fps_window_start: now,
            fps_frames: 0,
            poll_frame: 0,
            asset_backlog: PendingAssetUploadQueue::default(),
            immediate_backlog: PendingAssetUploadQueue::default(),
            upload_batch: Vec::new(),
            upload_bytes: Vec::new(),
        }
    }

    fn release_pending_uploads(&mut self, tuning: &Arc<RuntimeTuning>) {
        release_pending_uploads(tuning, &mut self.asset_backlog, &mut self.immediate_backlog);
    }

    fn collect_message(
        &mut self,
        message: RenderMessage,
        backlog_enabled: bool,
        max_pending: usize,
        max_pending_bytes: usize,
        should_render: &mut bool,
        stream_request_sender: &Sender<AssetStreamingRequest>,
        stats: &Arc<RendererStats>,
        tuning: &Arc<RuntimeTuning>,
        config: &mut RenderRuntimeConfig,
    ) -> bool {
        match message {
            RenderMessage::CreateMesh { .. }
            | RenderMessage::RemoveMesh { .. }
            | RenderMessage::CreateTexture { .. }
            | RenderMessage::CreateMaterial(_) => {
                let released_bytes = enqueue_asset_upload(
                    &mut self.asset_backlog,
                    &mut self.immediate_backlog,
                    message,
                    backlog_enabled,
                    max_pending,
                    max_pending_bytes,
                );
                if released_bytes > 0 {
                    tuning.release_asset_upload(released_bytes);
                }
                *should_render = true;
                true
            }
            RenderMessage::WindowRecreated { window, size } => {
                self.window = Some(Arc::clone(&window));
                let Some(renderer_ref) = self.renderer.as_mut() else {
                    warn!("window recreation requested without an active renderer");
                    return true;
                };
                let snapshot = renderer_ref.take_snapshot();
                self.surface_size = size;
                match create_renderer_for_window(
                    &window,
                    self.surface_size,
                    stream_request_sender,
                    stats,
                    tuning,
                    *config,
                ) {
                    Ok(mut renderer) => {
                        renderer.restore_snapshot(snapshot);
                        self.renderer = Some(renderer);
                        *should_render = true;
                    }
                    Err(err) => {
                        warn!("failed to recreate renderer: {err}");
                        renderer_ref.restore_snapshot(snapshot);
                    }
                }
                true
            }
            #[cfg(not(target_arch = "wasm32"))]
            RenderMessage::WindowRecreatedWithInit {
                window,
                size,
                render_init,
            } => {
                self.window = Some(Arc::clone(&window));
                let Some(renderer_ref) = self.renderer.as_mut() else {
                    warn!("window recreation requested without an active renderer");
                    return true;
                };
                let snapshot = renderer_ref.take_snapshot();
                self.surface_size = size;
                match create_renderer_from_init(
                    render_init,
                    self.surface_size,
                    stream_request_sender,
                    stats,
                    tuning,
                    *config,
                ) {
                    Ok(mut renderer) => {
                        renderer.restore_snapshot(snapshot);
                        self.renderer = Some(renderer);
                        *should_render = true;
                    }
                    Err(err) => {
                        warn!("failed to recreate renderer: {err}");
                        renderer_ref.restore_snapshot(snapshot);
                    }
                }
                true
            }
            RenderMessage::Resize(size) => {
                self.surface_size = PhysicalSize::new(size.width.max(1), size.height.max(1));
                *should_render = true;
                if let Some(renderer_ref) = self.renderer.as_mut() {
                    renderer_ref.process_message(RenderMessage::Resize(self.surface_size));
                }
                true
            }
            RenderMessage::Control(RenderControl::RecreateDevice {
                backend,
                binding_backend,
                allow_experimental_features,
            }) => {
                let Some(window) = self.window.as_ref().map(Arc::clone) else {
                    warn!("failed to recreate renderer: no window handle available");
                    return true;
                };
                let current_size = window.inner_size();
                let recreate_size =
                    PhysicalSize::new(current_size.width.max(1), current_size.height.max(1));
                self.surface_size = recreate_size;

                let previous_config = *config;
                let mut recreate_config = previous_config;
                recreate_config.wgpu_backend = backend;
                recreate_config.binding_backend = binding_backend;
                recreate_config.wgpu_experimental_features = allow_experimental_features;

                let Some(mut old_renderer) = self.renderer.take() else {
                    warn!("failed to recreate renderer: no active renderer available");
                    return true;
                };
                let mut snapshot = Some(old_renderer.take_device_recreate_snapshot());
                old_renderer.poll_device(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
                drop(old_renderer);

                match create_renderer_for_window(
                    &window,
                    recreate_size,
                    stream_request_sender,
                    stats,
                    tuning,
                    recreate_config,
                ) {
                    Ok(mut new_renderer) => {
                        new_renderer.restore_snapshot(snapshot.take().expect("snapshot available"));
                        self.renderer = Some(new_renderer);
                        *config = recreate_config;
                        *should_render = true;
                    }
                    Err(primary_err) => {
                        warn!(
                            "failed to recreate renderer with {} + {}: {primary_err}; restoring previous config",
                            backend.label(),
                            binding_backend.label()
                        );
                        match create_renderer_for_window(
                            &window,
                            recreate_size,
                            stream_request_sender,
                            stats,
                            tuning,
                            previous_config,
                        ) {
                            Ok(mut fallback_renderer) => {
                                fallback_renderer
                                    .restore_snapshot(snapshot.take().expect("snapshot available"));
                                self.renderer = Some(fallback_renderer);
                                *config = previous_config;
                                *should_render = true;
                            }
                            Err(fallback_err) => {
                                warn!(
                                    "failed to restore previous renderer config after recreate failure: {fallback_err}"
                                );
                            }
                        }
                    }
                }
                true
            }
            RenderMessage::Shutdown => {
                if let Some(renderer_ref) = self.renderer.as_mut() {
                    renderer_ref.process_message(RenderMessage::Shutdown);
                }
                false
            }
            other => {
                *should_render = true;
                if let Some(renderer_ref) = self.renderer.as_mut() {
                    renderer_ref.process_message(other);
                }
                true
            }
        }
    }

    fn render_frame(
        &mut self,
        frame_start: Instant,
        pending_render_messages: &mut VecDeque<RenderMessage>,
        pending_asset_messages: &mut VecDeque<RenderMessage>,
        control_receiver: &Receiver<RenderMessage>,
        render_receiver: &Receiver<RenderMessage>,
        asset_receiver: &Receiver<RenderMessage>,
        stream_request_sender: &Sender<AssetStreamingRequest>,
        stats: &Arc<RendererStats>,
        metrics: Option<&Arc<PerformanceMetrics>>,
        tuning: &Arc<RuntimeTuning>,
        profiling: &Arc<RuntimeProfiling>,
        config: &mut RenderRuntimeConfig,
    ) -> bool {
        let profiling_enabled = profiling.enabled.load(Ordering::Relaxed);
        let messages_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let mut should_render = false;

        let max_pending = tuning.max_pending_asset_uploads.load(Ordering::Relaxed);
        let max_pending_bytes = tuning.max_pending_asset_bytes.load(Ordering::Relaxed);
        let backlog_enabled = max_pending > 0 && max_pending_bytes > 0;
        if !backlog_enabled && !self.asset_backlog.is_empty() {
            self.asset_backlog.drain_release(tuning);
        }
        let base_asset_message_budget = asset_message_collection_budget(tuning);
        let mut processed_asset_messages = 0usize;

        while let Some(message) = pending_render_messages.pop_front() {
            if !self.collect_message(
                message,
                backlog_enabled,
                max_pending,
                max_pending_bytes,
                &mut should_render,
                stream_request_sender,
                stats,
                tuning,
                config,
            ) {
                self.release_pending_uploads(tuning);
                return false;
            }
        }

        loop {
            match control_receiver.try_recv() {
                Ok(message) => {
                    if !self.collect_message(
                        message,
                        backlog_enabled,
                        max_pending,
                        max_pending_bytes,
                        &mut should_render,
                        stream_request_sender,
                        stats,
                        tuning,
                        config,
                    ) {
                        self.release_pending_uploads(tuning);
                        return false;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.release_pending_uploads(tuning);
                    return false;
                }
            }
        }

        loop {
            match render_receiver.try_recv() {
                Ok(message) => {
                    if !self.collect_message(
                        message,
                        backlog_enabled,
                        max_pending,
                        max_pending_bytes,
                        &mut should_render,
                        stream_request_sender,
                        stats,
                        tuning,
                        config,
                    ) {
                        self.release_pending_uploads(tuning);
                        return false;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.release_pending_uploads(tuning);
                    return false;
                }
            }
        }

        let render_due = render_due_for_frame(frame_start, self.last_render, tuning);
        let base_uploads_per_frame = tuning.asset_uploads_per_frame.load(Ordering::Relaxed);
        let (asset_message_budget, uploads_per_frame) = asset_work_budgets_for_frame(
            base_asset_message_budget,
            base_uploads_per_frame,
            should_render || render_due,
        );

        while let Some(message) = pending_asset_messages.pop_front() {
            if processed_asset_messages >= asset_message_budget {
                pending_asset_messages.push_front(message);
                break;
            }
            if backlog_enabled && self.asset_backlog.len() >= max_pending {
                pending_asset_messages.push_front(message);
                break;
            }
            processed_asset_messages = processed_asset_messages.saturating_add(1);
            if !self.collect_message(
                message,
                backlog_enabled,
                max_pending,
                max_pending_bytes,
                &mut should_render,
                stream_request_sender,
                stats,
                tuning,
                config,
            ) {
                self.release_pending_uploads(tuning);
                return false;
            }
        }

        loop {
            if processed_asset_messages >= asset_message_budget {
                break;
            }
            if backlog_enabled && self.asset_backlog.len() >= max_pending {
                break;
            }
            match asset_receiver.try_recv() {
                Ok(message) => {
                    processed_asset_messages = processed_asset_messages.saturating_add(1);
                    if !self.collect_message(
                        message,
                        backlog_enabled,
                        max_pending,
                        max_pending_bytes,
                        &mut should_render,
                        stream_request_sender,
                        stats,
                        tuning,
                        config,
                    ) {
                        self.release_pending_uploads(tuning);
                        return false;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        let Some(renderer) = self.renderer.as_mut() else {
            warn!("renderer unavailable on native main thread");
            self.release_pending_uploads(tuning);
            return false;
        };

        let upload_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        self.upload_batch.clear();
        self.upload_bytes.clear();
        let mut uploads_this_frame = 0usize;
        while uploads_this_frame < uploads_per_frame {
            if let Some((message, bytes)) = self.immediate_backlog.pop_front() {
                self.upload_batch.push(message);
                self.upload_bytes.push(bytes);
                uploads_this_frame += 1;
                continue;
            }
            if let Some((message, bytes)) = self.asset_backlog.pop_front() {
                self.upload_batch.push(message);
                self.upload_bytes.push(bytes);
                uploads_this_frame += 1;
                continue;
            }
            break;
        }
        if !self.upload_batch.is_empty() {
            renderer.process_asset_batch(&mut self.upload_batch);
            for bytes in self.upload_bytes.drain(..) {
                tuning.release_asset_upload(bytes);
            }
            should_render = true;
        }
        if let Some(start) = upload_start {
            profiling
                .render_thread_upload_us
                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        if let Some(start) = messages_start {
            profiling
                .render_thread_messages_us
                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        if let Some(target_fps) = tuning.load_target_fps() {
            let frame_duration = Duration::from_secs_f32(1.0 / target_fps);
            let elapsed_since_last = frame_start.duration_since(self.last_render);
            if should_render || elapsed_since_last >= frame_duration {
                let render_start = if profiling_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                let render_result = renderer.render();
                if let Err(err) = render_result {
                    warn!("render error: {err}");
                } else {
                    publish_render_fps(
                        metrics,
                        &mut self.fps_window_start,
                        &mut self.fps_frames,
                        frame_start,
                    );
                }
                if let Some(start) = render_start {
                    profiling
                        .render_thread_render_us
                        .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
                self.last_render = frame_start;
            }
        } else {
            let render_start = if profiling_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let render_result = renderer.render();
            if let Err(err) = render_result {
                warn!("render error: {err}");
            } else {
                publish_render_fps(
                    metrics,
                    &mut self.fps_window_start,
                    &mut self.fps_frames,
                    frame_start,
                );
            }
            if let Some(start) = render_start {
                profiling
                    .render_thread_render_us
                    .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            self.last_render = frame_start;
        }

        let poll_interval = tuning.wgpu_poll_interval_frames.load(Ordering::Relaxed);
        let poll_mode = tuning.wgpu_poll_mode.load(Ordering::Relaxed);
        if poll_interval > 0 && poll_mode != 0 && (self.poll_frame % poll_interval == 0) {
            let poll_type = match poll_mode {
                2 => wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                },
                _ => wgpu::PollType::Poll,
            };
            renderer.poll_device(poll_type);
        }
        self.poll_frame = self.poll_frame.wrapping_add(1);

        if profiling_enabled {
            profiling
                .render_thread_frame_us
                .store(frame_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        true
    }
}

#[cfg(target_arch = "wasm32")]
type RenderInitResult = Result<WebRenderState, String>;

#[cfg(target_arch = "wasm32")]
struct WebRenderState {
    renderer: crate::graphics::renderer::GraphRenderer,
    surface_size: PhysicalSize<u32>,
    last_render: Instant,
    fps_window_start: Instant,
    fps_frames: u32,
    poll_frame: u32,
    asset_backlog: PendingAssetUploadQueue,
    immediate_backlog: PendingAssetUploadQueue,
    upload_batch: Vec<RenderMessage>,
    upload_bytes: Vec<usize>,
}

#[cfg(target_arch = "wasm32")]
impl WebRenderState {
    fn new(
        renderer: crate::graphics::renderer::GraphRenderer,
        surface_size: PhysicalSize<u32>,
    ) -> Self {
        let now = Instant::now();
        Self {
            renderer,
            surface_size,
            last_render: now,
            fps_window_start: now,
            fps_frames: 0,
            poll_frame: 0,
            asset_backlog: PendingAssetUploadQueue::default(),
            immediate_backlog: PendingAssetUploadQueue::default(),
            upload_batch: Vec::new(),
            upload_bytes: Vec::new(),
        }
    }

    fn collect_message(
        &mut self,
        message: RenderMessage,
        backlog_enabled: bool,
        max_pending: usize,
        max_pending_bytes: usize,
        should_render: &mut bool,
        tuning: &Arc<RuntimeTuning>,
    ) -> bool {
        match message {
            RenderMessage::CreateMesh { .. }
            | RenderMessage::RemoveMesh { .. }
            | RenderMessage::CreateTexture { .. }
            | RenderMessage::CreateMaterial(_) => {
                let released_bytes = enqueue_asset_upload(
                    &mut self.asset_backlog,
                    &mut self.immediate_backlog,
                    message,
                    backlog_enabled,
                    max_pending,
                    max_pending_bytes,
                );
                if released_bytes > 0 {
                    tuning.release_asset_upload(released_bytes);
                }
                *should_render = true;
                true
            }
            RenderMessage::WindowRecreated { size, .. } => {
                self.surface_size = size;
                self.renderer.process_message(RenderMessage::Resize(size));
                *should_render = true;
                true
            }
            RenderMessage::Shutdown => {
                self.renderer.process_message(RenderMessage::Shutdown);
                false
            }
            other => {
                *should_render = true;
                self.renderer.process_message(other);
                true
            }
        }
    }

    fn render_frame(
        &mut self,
        frame_start: Instant,
        pending_render_messages: &mut VecDeque<RenderMessage>,
        pending_asset_messages: &mut VecDeque<RenderMessage>,
        control_receiver: &Receiver<RenderMessage>,
        render_receiver: &Receiver<RenderMessage>,
        asset_receiver: &Receiver<RenderMessage>,
        metrics: Option<&Arc<PerformanceMetrics>>,
        tuning: &Arc<RuntimeTuning>,
        profiling: &Arc<RuntimeProfiling>,
    ) -> bool {
        let profiling_enabled = profiling.enabled.load(Ordering::Relaxed);
        let messages_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let mut should_render = false;

        let max_pending = tuning.max_pending_asset_uploads.load(Ordering::Relaxed);
        let max_pending_bytes = tuning.max_pending_asset_bytes.load(Ordering::Relaxed);
        let backlog_enabled = max_pending > 0 && max_pending_bytes > 0;
        if !backlog_enabled && !self.asset_backlog.is_empty() {
            self.asset_backlog.drain_release(tuning);
        }

        while let Some(message) = pending_render_messages.pop_front() {
            if !self.collect_message(
                message,
                backlog_enabled,
                max_pending,
                max_pending_bytes,
                &mut should_render,
                tuning,
            ) {
                release_pending_uploads(
                    tuning,
                    &mut self.asset_backlog,
                    &mut self.immediate_backlog,
                );
                return false;
            }
        }

        loop {
            match control_receiver.try_recv() {
                Ok(message) => {
                    if !self.collect_message(
                        message,
                        backlog_enabled,
                        max_pending,
                        max_pending_bytes,
                        &mut should_render,
                        tuning,
                    ) {
                        release_pending_uploads(
                            tuning,
                            &mut self.asset_backlog,
                            &mut self.immediate_backlog,
                        );
                        return false;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    release_pending_uploads(
                        tuning,
                        &mut self.asset_backlog,
                        &mut self.immediate_backlog,
                    );
                    return false;
                }
            }
        }

        loop {
            match render_receiver.try_recv() {
                Ok(message) => {
                    if !self.collect_message(
                        message,
                        backlog_enabled,
                        max_pending,
                        max_pending_bytes,
                        &mut should_render,
                        tuning,
                    ) {
                        release_pending_uploads(
                            tuning,
                            &mut self.asset_backlog,
                            &mut self.immediate_backlog,
                        );
                        return false;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    release_pending_uploads(
                        tuning,
                        &mut self.asset_backlog,
                        &mut self.immediate_backlog,
                    );
                    return false;
                }
            }
        }

        loop {
            if backlog_enabled && self.asset_backlog.len() >= max_pending {
                break;
            }
            let Some(message) = pending_asset_messages.pop_front() else {
                break;
            };
            if !self.collect_message(
                message,
                backlog_enabled,
                max_pending,
                max_pending_bytes,
                &mut should_render,
                tuning,
            ) {
                release_pending_uploads(
                    tuning,
                    &mut self.asset_backlog,
                    &mut self.immediate_backlog,
                );
                return false;
            }
        }

        loop {
            if backlog_enabled && self.asset_backlog.len() >= max_pending {
                break;
            }
            match asset_receiver.try_recv() {
                Ok(message) => {
                    if !self.collect_message(
                        message,
                        backlog_enabled,
                        max_pending,
                        max_pending_bytes,
                        &mut should_render,
                        tuning,
                    ) {
                        release_pending_uploads(
                            tuning,
                            &mut self.asset_backlog,
                            &mut self.immediate_backlog,
                        );
                        return false;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        let uploads_per_frame = tuning.asset_uploads_per_frame.load(Ordering::Relaxed);
        let upload_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        self.upload_batch.clear();
        self.upload_bytes.clear();
        let mut uploads_this_frame = 0usize;
        while uploads_this_frame < uploads_per_frame {
            if let Some((message, bytes)) = self.immediate_backlog.pop_front() {
                self.upload_batch.push(message);
                self.upload_bytes.push(bytes);
                uploads_this_frame += 1;
                continue;
            }
            if let Some((message, bytes)) = self.asset_backlog.pop_front() {
                self.upload_batch.push(message);
                self.upload_bytes.push(bytes);
                uploads_this_frame += 1;
                continue;
            }
            break;
        }
        if !self.upload_batch.is_empty() {
            self.renderer.process_asset_batch(&mut self.upload_batch);
            for bytes in self.upload_bytes.drain(..) {
                tuning.release_asset_upload(bytes);
            }
            should_render = true;
        }
        if let Some(start) = upload_start {
            profiling
                .render_thread_upload_us
                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        if let Some(start) = messages_start {
            profiling
                .render_thread_messages_us
                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        if let Some(target_fps) = tuning.load_target_fps() {
            let frame_duration = Duration::from_secs_f32(1.0 / target_fps);
            let elapsed_since_last = frame_start.duration_since(self.last_render);
            if should_render || elapsed_since_last >= frame_duration {
                let render_start = if profiling_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                let render_result = self.renderer.render();
                if let Err(err) = render_result {
                    warn!("render error: {err}");
                } else {
                    publish_render_fps(
                        metrics,
                        &mut self.fps_window_start,
                        &mut self.fps_frames,
                        frame_start,
                    );
                }
                if let Some(start) = render_start {
                    profiling
                        .render_thread_render_us
                        .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
                self.last_render = frame_start;
            }
        } else {
            let render_start = if profiling_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let render_result = self.renderer.render();
            if let Err(err) = render_result {
                warn!("render error: {err}");
            } else {
                publish_render_fps(
                    metrics,
                    &mut self.fps_window_start,
                    &mut self.fps_frames,
                    frame_start,
                );
            }
            if let Some(start) = render_start {
                profiling
                    .render_thread_render_us
                    .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            self.last_render = frame_start;
        }

        let poll_interval = tuning.wgpu_poll_interval_frames.load(Ordering::Relaxed);
        let poll_mode = tuning.wgpu_poll_mode.load(Ordering::Relaxed);
        if poll_interval > 0 && poll_mode != 0 && (self.poll_frame % poll_interval == 0) {
            let poll_type = match poll_mode {
                2 => wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                },
                _ => wgpu::PollType::Poll,
            };
            self.renderer.poll_device(poll_type);
        }
        self.poll_frame = self.poll_frame.wrapping_add(1);

        if profiling_enabled {
            profiling
                .render_thread_frame_us
                .store(frame_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        true
    }
}

#[cfg(target_arch = "wasm32")]
impl RenderExtension {
    fn pump_channels_on_main_thread(&mut self) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }

        self.handle_render_init();

        if self.render_state.is_none() {
            self.collect_messages_before_renderer_ready();
            if !self.running.load(Ordering::Acquire) {
                return;
            }

            if self.render_init_receiver.is_none() {
                if let Some((window, size)) = self.pending_bootstrap.clone() {
                    let retry_ready = self
                        .next_renderer_init_retry_at
                        .map(|deadline| Instant::now() >= deadline)
                        .unwrap_or(true);
                    if retry_ready {
                        self.enqueue_renderer_init(window, size);
                    }
                }
            }
            return;
        }

        let frame_start = Instant::now();
        let keep_alive = self.render_state.as_mut().is_some_and(|state| {
            state.render_frame(
                frame_start,
                &mut self.pending_render_messages,
                &mut self.pending_asset_messages,
                &self.control_receiver,
                &self.render_receiver,
                &self.asset_receiver,
                self.metrics.as_ref(),
                &self.tuning,
                &self.profiling,
            )
        });

        if !keep_alive {
            self.running.store(false, Ordering::Release);
            self.release_queued_messages();
            self.render_state = None;
            self.render_init_receiver = None;
            self.pending_bootstrap = None;
        }
    }

    fn collect_messages_before_renderer_ready(&mut self) {
        loop {
            match self.control_receiver.try_recv() {
                Ok(RenderMessage::WindowRecreated { window, size }) => {
                    self.pending_bootstrap = Some((window, size));
                    self.next_renderer_init_retry_at = None;
                }
                Ok(RenderMessage::Shutdown) => {
                    self.running.store(false, Ordering::Release);
                    self.release_queued_messages();
                    return;
                }
                Ok(message) => {
                    self.pending_render_messages.push_back(message);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.running.store(false, Ordering::Release);
                    self.release_queued_messages();
                    return;
                }
            }
        }

        loop {
            match self.render_receiver.try_recv() {
                Ok(RenderMessage::WindowRecreated { window, size }) => {
                    self.pending_bootstrap = Some((window, size));
                    self.next_renderer_init_retry_at = None;
                }
                Ok(RenderMessage::Shutdown) => {
                    self.running.store(false, Ordering::Release);
                    self.release_queued_messages();
                    return;
                }
                Ok(message) => {
                    self.pending_render_messages.push_back(message);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.running.store(false, Ordering::Release);
                    self.release_queued_messages();
                    return;
                }
            }
        }

        loop {
            match self.asset_receiver.try_recv() {
                Ok(RenderMessage::Shutdown) => {
                    self.running.store(false, Ordering::Release);
                    self.release_queued_messages();
                    return;
                }
                Ok(message) => {
                    self.pending_asset_messages.push_back(message);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }

    fn enqueue_renderer_init(&mut self, window: Arc<Window>, size: PhysicalSize<u32>) {
        if self.render_state.is_some() || self.render_init_receiver.is_some() {
            return;
        }

        let render_size = PhysicalSize::new(size.width.max(1), size.height.max(1));
        let (sender, receiver) = bounded(1);
        self.render_init_receiver = Some(receiver);

        let stream_request_sender = self.stream_request_sender.clone();
        let stats = Arc::clone(&self.stats);
        let target_tickrate = self.tuning.load_target_tickrate();
        let config = self.config;

        spawn_local(async move {
            let init_with_backend = |backend: WgpuBackend| {
                let window = Arc::clone(&window);
                let stream_request_sender = stream_request_sender.clone();
                let stats = Arc::clone(&stats);
                async move {
                    let instance = make_wgpu_instance(backend).await;
                    let surface = create_surface(&instance, &window)?;
                    initialize_renderer(
                        instance,
                        surface,
                        render_size,
                        target_tickrate,
                        stream_request_sender,
                        stats,
                        config.wgpu_experimental_features,
                        backend,
                        config.binding_backend,
                    )
                    .await
                    .map_err(|err| err.to_string())
                    .map(|renderer| WebRenderState::new(renderer, render_size))
                }
            };

            let init_result = match init_with_backend(config.wgpu_backend).await {
                Ok(state) => Ok(state),
                Err(primary_err) if config.wgpu_backend != WgpuBackend::Gl => {
                    warn!(
                        "renderer init failed with {} backend: {}; retrying with OpenGL",
                        config.wgpu_backend.label(),
                        primary_err
                    );
                    match init_with_backend(WgpuBackend::Gl).await {
                        Ok(state) => Ok(state),
                        Err(fallback_err) => Err(format!(
                            "{primary_err}; OpenGL fallback failed: {fallback_err}"
                        )),
                    }
                }
                Err(err) => Err(err),
            };

            let _ = sender.send(init_result);
        });
    }

    fn handle_render_init(&mut self) {
        let Some(receiver) = &self.render_init_receiver else {
            return;
        };
        match receiver.try_recv() {
            Ok(Ok(mut state)) => {
                if let Some((_, size)) = self.pending_bootstrap.take() {
                    state.renderer.process_message(RenderMessage::Resize(size));
                }
                self.render_state = Some(state);
                self.render_init_receiver = None;
                self.next_renderer_init_retry_at = None;
                self.renderer_init_failures = 0;
            }
            Ok(Err(err)) => {
                warn!("failed to initialize renderer: {err}");
                web_sys::console::error_1(
                    &format!("helmer_render: failed to initialize renderer: {err}").into(),
                );
                self.render_init_receiver = None;
                self.renderer_init_failures = self.renderer_init_failures.saturating_add(1);
                let shift = self.renderer_init_failures.saturating_sub(1).min(6);
                let backoff_ms = 100u64.saturating_mul(1u64 << shift);
                self.next_renderer_init_retry_at =
                    Some(Instant::now() + Duration::from_millis(backoff_ms));
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                warn!("renderer init channel disconnected");
                self.render_init_receiver = None;
                self.renderer_init_failures = self.renderer_init_failures.saturating_add(1);
                let shift = self.renderer_init_failures.saturating_sub(1).min(6);
                let backoff_ms = 100u64.saturating_mul(1u64 << shift);
                self.next_renderer_init_retry_at =
                    Some(Instant::now() + Duration::from_millis(backoff_ms));
            }
        }
    }

    fn release_queued_messages(&mut self) {
        for message in self.pending_asset_messages.drain(..) {
            let bytes = render_message_payload_bytes(&message);
            if bytes > 0 {
                self.tuning.release_asset_upload(bytes);
            }
        }
        for message in self.pending_render_messages.drain(..) {
            let bytes = render_message_payload_bytes(&message);
            if bytes > 0 {
                self.tuning.release_asset_upload(bytes);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RenderExtension {
    fn release_queued_messages_native(&mut self) {
        if let Some(state) = self.native_render_state.as_mut() {
            state.release_pending_uploads(&self.tuning);
        }
        for message in self.native_pending_asset_messages.drain(..) {
            let bytes = render_message_payload_bytes(&message);
            if bytes > 0 {
                self.tuning.release_asset_upload(bytes);
            }
        }
        for message in self.native_pending_render_messages.drain(..) {
            let bytes = render_message_payload_bytes(&message);
            if bytes > 0 {
                self.tuning.release_asset_upload(bytes);
            }
        }
    }

    fn collect_messages_before_renderer_ready_native(&mut self) {
        loop {
            match self.control_receiver.try_recv() {
                Ok(RenderMessage::WindowRecreated { window, size }) => {
                    self.native_pending_bootstrap =
                        Some(NativeRendererBootstrap::Window { window, size });
                }
                Ok(RenderMessage::WindowRecreatedWithInit {
                    window,
                    size,
                    render_init,
                }) => {
                    self.native_pending_bootstrap = Some(NativeRendererBootstrap::Init {
                        window,
                        render_init,
                        size,
                    });
                }
                Ok(RenderMessage::Shutdown) => {
                    self.running.store(false, Ordering::Release);
                    self.release_queued_messages_native();
                    return;
                }
                Ok(message) => {
                    self.native_pending_render_messages.push_back(message);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.running.store(false, Ordering::Release);
                    self.release_queued_messages_native();
                    return;
                }
            }
        }

        loop {
            match self.render_receiver.try_recv() {
                Ok(RenderMessage::WindowRecreated { window, size }) => {
                    self.native_pending_bootstrap =
                        Some(NativeRendererBootstrap::Window { window, size });
                }
                Ok(RenderMessage::WindowRecreatedWithInit {
                    window,
                    size,
                    render_init,
                }) => {
                    self.native_pending_bootstrap = Some(NativeRendererBootstrap::Init {
                        window,
                        render_init,
                        size,
                    });
                }
                Ok(RenderMessage::Shutdown) => {
                    self.running.store(false, Ordering::Release);
                    self.release_queued_messages_native();
                    return;
                }
                Ok(message) => {
                    self.native_pending_render_messages.push_back(message);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.running.store(false, Ordering::Release);
                    self.release_queued_messages_native();
                    return;
                }
            }
        }

        loop {
            match self.asset_receiver.try_recv() {
                Ok(RenderMessage::Shutdown) => {
                    self.running.store(false, Ordering::Release);
                    self.release_queued_messages_native();
                    return;
                }
                Ok(message) => {
                    self.native_pending_asset_messages.push_back(message);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }

    fn pump_channels_on_main_thread_native(&mut self) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }

        if self.native_render_state.is_none() {
            self.collect_messages_before_renderer_ready_native();
            if !self.running.load(Ordering::Acquire) {
                return;
            }
            if let Some(bootstrap) = self.native_pending_bootstrap.take() {
                match bootstrap {
                    NativeRendererBootstrap::Window { window, size } => {
                        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));
                        match create_renderer_for_window(
                            &window,
                            size,
                            &self.stream_request_sender,
                            &self.stats,
                            &self.tuning,
                            self.config,
                        ) {
                            Ok(renderer) => {
                                self.native_render_state = Some(NativeInlineRenderState::new(
                                    renderer,
                                    Some(window),
                                    size,
                                ));
                            }
                            Err(err) => {
                                warn!("failed to initialize renderer: {err}");
                                self.native_pending_bootstrap =
                                    Some(NativeRendererBootstrap::Window { window, size });
                            }
                        }
                    }
                    NativeRendererBootstrap::Init {
                        window,
                        render_init,
                        size,
                    } => {
                        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));
                        match create_renderer_from_init(
                            render_init,
                            size,
                            &self.stream_request_sender,
                            &self.stats,
                            &self.tuning,
                            self.config,
                        ) {
                            Ok(renderer) => {
                                self.native_render_state = Some(NativeInlineRenderState::new(
                                    renderer,
                                    Some(window),
                                    size,
                                ));
                            }
                            Err(err) => {
                                warn!("failed to initialize renderer: {err}");
                            }
                        }
                    }
                }
            }
            return;
        }

        let frame_start = Instant::now();
        let keep_alive = if let Some(state) = self.native_render_state.as_mut() {
            state.render_frame(
                frame_start,
                &mut self.native_pending_render_messages,
                &mut self.native_pending_asset_messages,
                &self.control_receiver,
                &self.render_receiver,
                &self.asset_receiver,
                &self.stream_request_sender,
                &self.stats,
                self.metrics.as_ref(),
                &self.tuning,
                &self.profiling,
                &mut self.config,
            )
        } else {
            false
        };

        if !keep_alive {
            self.running.store(false, Ordering::Release);
            self.release_queued_messages_native();
            self.native_render_state = None;
            self.native_pending_bootstrap = None;
        }
    }

    fn spawn_render_thread(&self, ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        let running = Arc::clone(&self.running);
        let control_receiver = self.control_receiver.clone();
        let render_receiver = self.render_receiver.clone();
        let asset_receiver = self.asset_receiver.clone();
        let stream_request_sender = self.stream_request_sender.clone();
        let stats = Arc::clone(&self.stats);
        let metrics = self.metrics.clone();
        let tuning = Arc::clone(&self.tuning);
        let profiling = Arc::clone(&self.profiling);
        let config = self.config;

        ctx.threads().spawn_named("helmer-render", move || {
            run_render_thread(
                running,
                control_receiver,
                render_receiver,
                asset_receiver,
                stream_request_sender,
                stats,
                metrics,
                tuning,
                profiling,
                config,
            );
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_render_thread(
    running: Arc<AtomicBool>,
    control_receiver: Receiver<RenderMessage>,
    render_receiver: Receiver<RenderMessage>,
    asset_receiver: Receiver<RenderMessage>,
    stream_request_sender: Sender<AssetStreamingRequest>,
    stats: Arc<RendererStats>,
    metrics: Option<Arc<PerformanceMetrics>>,
    tuning: Arc<RuntimeTuning>,
    profiling: Arc<RuntimeProfiling>,
    config: RenderRuntimeConfig,
) {
    let _high_resolution_timer = helmer::runtime::HighResolutionTimerGuard::new(1);

    let mut renderer: Option<crate::graphics::renderer::GraphRenderer> = None;
    let mut active_window: Option<Arc<Window>> = None;
    let mut active_config = config;
    let mut surface_size = PhysicalSize::new(1, 1);
    let mut last_render = Instant::now();
    let mut fps_window_start = Instant::now();
    let mut fps_frames: u32 = 0;
    let mut poll_frame: u32 = 0;

    let mut asset_backlog = PendingAssetUploadQueue::default();
    let mut immediate_backlog = PendingAssetUploadQueue::default();
    let mut upload_batch: Vec<RenderMessage> = Vec::new();
    let mut upload_bytes: Vec<usize> = Vec::new();
    let mut pending_bootstrap_render_messages: VecDeque<RenderMessage> = VecDeque::new();

    'render_loop: while running.load(Ordering::Relaxed) {
        if renderer.is_none() {
            let Some(bootstrap) = wait_for_renderer_bootstrap(
                &running,
                &control_receiver,
                &render_receiver,
                &mut pending_bootstrap_render_messages,
            ) else {
                break;
            };
            match bootstrap {
                NativeRendererBootstrap::Window { window, size } => {
                    active_window = Some(Arc::clone(&window));
                    surface_size = size;
                    match create_renderer_for_window(
                        &window,
                        surface_size,
                        &stream_request_sender,
                        &stats,
                        &tuning,
                        active_config,
                    ) {
                        Ok(new_renderer) => {
                            renderer = Some(new_renderer);
                            let now = Instant::now();
                            last_render = now;
                            fps_window_start = now;
                            fps_frames = 0;
                        }
                        Err(err) => {
                            warn!("failed to initialize renderer: {err}");
                            thread::sleep(Duration::from_millis(16));
                        }
                    }
                }
                NativeRendererBootstrap::Init {
                    window,
                    render_init,
                    size,
                } => {
                    active_window = Some(window);
                    surface_size = size;
                    match create_renderer_from_init(
                        render_init,
                        surface_size,
                        &stream_request_sender,
                        &stats,
                        &tuning,
                        active_config,
                    ) {
                        Ok(new_renderer) => {
                            renderer = Some(new_renderer);
                            let now = Instant::now();
                            last_render = now;
                            fps_window_start = now;
                            fps_frames = 0;
                        }
                        Err(err) => {
                            warn!("failed to initialize renderer: {err}");
                            thread::sleep(Duration::from_millis(16));
                        }
                    }
                }
            }
            continue;
        }

        let frame_start = Instant::now();
        let profiling_enabled = profiling.enabled.load(Ordering::Relaxed);
        let messages_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let mut should_render = false;

        let max_pending = tuning.max_pending_asset_uploads.load(Ordering::Relaxed);
        let max_pending_bytes = tuning.max_pending_asset_bytes.load(Ordering::Relaxed);
        let backlog_enabled = max_pending > 0 && max_pending_bytes > 0;
        if !backlog_enabled && !asset_backlog.is_empty() {
            asset_backlog.drain_release(&tuning);
        }
        let base_asset_message_budget = asset_message_collection_budget(&tuning);
        let mut processed_asset_messages = 0usize;

        if !drain_control_messages(&control_receiver, &mut pending_bootstrap_render_messages) {
            release_pending_uploads(&tuning, &mut asset_backlog, &mut immediate_backlog);
            return;
        }

        loop {
            let message = if let Some(message) = pending_bootstrap_render_messages.pop_front() {
                Some(message)
            } else {
                match render_receiver.try_recv() {
                    Ok(message) => Some(message),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => {
                        release_pending_uploads(
                            &tuning,
                            &mut asset_backlog,
                            &mut immediate_backlog,
                        );
                        return;
                    }
                }
            };
            let Some(message) = message else {
                break;
            };
            match message {
                RenderMessage::CreateMesh { .. }
                | RenderMessage::RemoveMesh { .. }
                | RenderMessage::CreateTexture { .. }
                | RenderMessage::CreateMaterial(_) => {
                    let released_bytes = enqueue_asset_upload(
                        &mut asset_backlog,
                        &mut immediate_backlog,
                        message,
                        backlog_enabled,
                        max_pending,
                        max_pending_bytes,
                    );
                    if released_bytes > 0 {
                        tuning.release_asset_upload(released_bytes);
                    }
                    should_render = true;
                }
                RenderMessage::WindowRecreated { window, size } => {
                    if let Some(renderer_ref) = renderer.as_mut() {
                        active_window = Some(Arc::clone(&window));
                        let snapshot = renderer_ref.take_snapshot();
                        surface_size = size;
                        match create_renderer_for_window(
                            &window,
                            surface_size,
                            &stream_request_sender,
                            &stats,
                            &tuning,
                            active_config,
                        ) {
                            Ok(mut new_renderer) => {
                                new_renderer.restore_snapshot(snapshot);
                                *renderer_ref = new_renderer;
                                should_render = true;
                            }
                            Err(err) => {
                                warn!("failed to recreate renderer: {err}");
                                renderer_ref.restore_snapshot(snapshot);
                            }
                        }
                    } else {
                        warn!("window recreation requested without an active renderer");
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                RenderMessage::WindowRecreatedWithInit {
                    window,
                    size,
                    render_init,
                } => {
                    if let Some(renderer_ref) = renderer.as_mut() {
                        active_window = Some(window);
                        let snapshot = renderer_ref.take_snapshot();
                        surface_size = size;
                        match create_renderer_from_init(
                            render_init,
                            surface_size,
                            &stream_request_sender,
                            &stats,
                            &tuning,
                            active_config,
                        ) {
                            Ok(mut new_renderer) => {
                                new_renderer.restore_snapshot(snapshot);
                                *renderer_ref = new_renderer;
                                should_render = true;
                            }
                            Err(err) => {
                                warn!("failed to recreate renderer: {err}");
                                renderer_ref.restore_snapshot(snapshot);
                            }
                        }
                    } else {
                        warn!("window recreation requested without an active renderer");
                    }
                }
                RenderMessage::Resize(size) => {
                    surface_size = PhysicalSize::new(size.width.max(1), size.height.max(1));
                    should_render = true;
                    if let Some(renderer_ref) = renderer.as_mut() {
                        renderer_ref.process_message(RenderMessage::Resize(surface_size));
                    }
                }
                RenderMessage::Control(RenderControl::RecreateDevice {
                    backend,
                    binding_backend,
                    allow_experimental_features,
                }) => {
                    let Some(window) = active_window.as_ref().map(Arc::clone) else {
                        warn!("failed to recreate renderer: no window handle available");
                        continue;
                    };
                    let current_size = window.inner_size();
                    let recreate_size =
                        PhysicalSize::new(current_size.width.max(1), current_size.height.max(1));

                    let previous_config = active_config;
                    let mut recreate_config = previous_config;
                    recreate_config.wgpu_backend = backend;
                    recreate_config.binding_backend = binding_backend;
                    recreate_config.wgpu_experimental_features = allow_experimental_features;

                    let Some(mut old_renderer) = renderer.take() else {
                        warn!("failed to recreate renderer: no active renderer available");
                        continue;
                    };
                    let mut snapshot = Some(old_renderer.take_device_recreate_snapshot());
                    old_renderer.poll_device(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    });
                    drop(old_renderer);

                    match create_renderer_for_window(
                        &window,
                        recreate_size,
                        &stream_request_sender,
                        &stats,
                        &tuning,
                        recreate_config,
                    ) {
                        Ok(mut new_renderer) => {
                            new_renderer
                                .restore_snapshot(snapshot.take().expect("snapshot available"));
                            renderer = Some(new_renderer);
                            active_config = recreate_config;
                            should_render = true;
                        }
                        Err(primary_err) => {
                            warn!(
                                "failed to recreate renderer with {} + {}: {primary_err}; restoring previous config",
                                backend.label(),
                                binding_backend.label()
                            );
                            match create_renderer_for_window(
                                &window,
                                recreate_size,
                                &stream_request_sender,
                                &stats,
                                &tuning,
                                previous_config,
                            ) {
                                Ok(mut fallback_renderer) => {
                                    fallback_renderer.restore_snapshot(
                                        snapshot.take().expect("snapshot available"),
                                    );
                                    renderer = Some(fallback_renderer);
                                    active_config = previous_config;
                                    should_render = true;
                                }
                                Err(fallback_err) => {
                                    warn!(
                                        "failed to restore previous renderer config after recreate failure: {fallback_err}; shutting down render thread"
                                    );
                                    release_pending_uploads(
                                        &tuning,
                                        &mut asset_backlog,
                                        &mut immediate_backlog,
                                    );
                                    return;
                                }
                            }
                        }
                    }
                }
                RenderMessage::Shutdown => {
                    if let Some(renderer_ref) = renderer.as_mut() {
                        renderer_ref.process_message(RenderMessage::Shutdown);
                    }
                    release_pending_uploads(&tuning, &mut asset_backlog, &mut immediate_backlog);
                    return;
                }
                other => {
                    should_render = true;
                    if let Some(renderer_ref) = renderer.as_mut() {
                        renderer_ref.process_message(other);
                    }
                }
            }
            if !drain_control_messages(&control_receiver, &mut pending_bootstrap_render_messages) {
                release_pending_uploads(&tuning, &mut asset_backlog, &mut immediate_backlog);
                return;
            }
        }

        let render_due = render_due_for_frame(frame_start, last_render, &tuning);
        let base_uploads_per_frame = tuning.asset_uploads_per_frame.load(Ordering::Relaxed);
        let (asset_message_budget, uploads_per_frame) = asset_work_budgets_for_frame(
            base_asset_message_budget,
            base_uploads_per_frame,
            should_render || render_due,
        );

        loop {
            if processed_asset_messages >= asset_message_budget {
                break;
            }
            if backlog_enabled && asset_backlog.len() >= max_pending {
                break;
            }

            match asset_receiver.try_recv() {
                Ok(message) => match message {
                    RenderMessage::CreateMesh { .. }
                    | RenderMessage::RemoveMesh { .. }
                    | RenderMessage::CreateTexture { .. }
                    | RenderMessage::CreateMaterial(_) => {
                        processed_asset_messages = processed_asset_messages.saturating_add(1);
                        let released_bytes = enqueue_asset_upload(
                            &mut asset_backlog,
                            &mut immediate_backlog,
                            message,
                            backlog_enabled,
                            max_pending,
                            max_pending_bytes,
                        );
                        if released_bytes > 0 {
                            tuning.release_asset_upload(released_bytes);
                        }
                        should_render = true;
                    }
                    RenderMessage::Shutdown => {
                        if let Some(renderer_ref) = renderer.as_mut() {
                            renderer_ref.process_message(RenderMessage::Shutdown);
                        }
                        release_pending_uploads(
                            &tuning,
                            &mut asset_backlog,
                            &mut immediate_backlog,
                        );
                        return;
                    }
                    other => {
                        should_render = true;
                        if let Some(renderer_ref) = renderer.as_mut() {
                            renderer_ref.process_message(other);
                        }
                    }
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
            if !drain_control_messages(&control_receiver, &mut pending_bootstrap_render_messages) {
                release_pending_uploads(&tuning, &mut asset_backlog, &mut immediate_backlog);
                return;
            }
            if !pending_bootstrap_render_messages.is_empty() {
                if let Some(start) = messages_start {
                    profiling
                        .render_thread_messages_us
                        .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
                continue 'render_loop;
            }
        }

        let Some(renderer) = renderer.as_mut() else {
            warn!("renderer unavailable on render thread; shutting down");
            release_pending_uploads(&tuning, &mut asset_backlog, &mut immediate_backlog);
            return;
        };

        let upload_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        upload_batch.clear();
        upload_bytes.clear();
        let mut uploads_this_frame = 0usize;
        while uploads_this_frame < uploads_per_frame {
            if let Some((message, bytes)) = immediate_backlog.pop_front() {
                upload_batch.push(message);
                upload_bytes.push(bytes);
                uploads_this_frame += 1;
                continue;
            }
            if let Some((message, bytes)) = asset_backlog.pop_front() {
                upload_batch.push(message);
                upload_bytes.push(bytes);
                uploads_this_frame += 1;
                continue;
            }
            break;
        }
        if !upload_batch.is_empty() {
            renderer.process_asset_batch(&mut upload_batch);
            for bytes in upload_bytes.drain(..) {
                tuning.release_asset_upload(bytes);
            }
            should_render = true;
        }
        if let Some(start) = upload_start {
            profiling
                .render_thread_upload_us
                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        if !drain_control_messages(&control_receiver, &mut pending_bootstrap_render_messages) {
            release_pending_uploads(&tuning, &mut asset_backlog, &mut immediate_backlog);
            return;
        }
        if !pending_bootstrap_render_messages.is_empty() {
            if let Some(start) = messages_start {
                profiling
                    .render_thread_messages_us
                    .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            continue;
        }

        if let Some(start) = messages_start {
            profiling
                .render_thread_messages_us
                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        if let Some(target_fps) = tuning.load_target_fps() {
            let frame_duration = Duration::from_secs_f32(1.0 / target_fps);
            let elapsed_since_last = frame_start.duration_since(last_render);
            if should_render || elapsed_since_last >= frame_duration {
                let render_start = if profiling_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                let render_result = renderer.render();
                if let Err(err) = render_result {
                    warn!("render error: {err}");
                } else {
                    publish_render_fps(
                        metrics.as_ref(),
                        &mut fps_window_start,
                        &mut fps_frames,
                        frame_start,
                    );
                }
                if let Some(start) = render_start {
                    profiling
                        .render_thread_render_us
                        .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
                last_render = frame_start;
            }
            let elapsed = frame_start.elapsed();
            if elapsed < frame_duration {
                thread::sleep(frame_duration - elapsed);
            }
        } else {
            let render_start = if profiling_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let render_result = renderer.render();
            if let Err(err) = render_result {
                warn!("render error: {err}");
            } else {
                publish_render_fps(
                    metrics.as_ref(),
                    &mut fps_window_start,
                    &mut fps_frames,
                    frame_start,
                );
            }
            if let Some(start) = render_start {
                profiling
                    .render_thread_render_us
                    .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            last_render = frame_start;
        }

        let poll_interval = tuning.wgpu_poll_interval_frames.load(Ordering::Relaxed);
        let poll_mode = tuning.wgpu_poll_mode.load(Ordering::Relaxed);
        if poll_interval > 0 && poll_mode != 0 && (poll_frame % poll_interval == 0) {
            let poll_type = match poll_mode {
                2 => wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                },
                _ => wgpu::PollType::Poll,
            };
            renderer.poll_device(poll_type);
        }
        poll_frame = poll_frame.wrapping_add(1);

        if profiling_enabled {
            profiling
                .render_thread_frame_us
                .store(frame_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }

    release_pending_uploads(&tuning, &mut asset_backlog, &mut immediate_backlog);
}

#[cfg(not(target_arch = "wasm32"))]
fn wait_for_renderer_bootstrap(
    running: &Arc<AtomicBool>,
    control_receiver: &Receiver<RenderMessage>,
    render_receiver: &Receiver<RenderMessage>,
    pending_messages: &mut VecDeque<RenderMessage>,
) -> Option<NativeRendererBootstrap> {
    while running.load(Ordering::Relaxed) {
        loop {
            match control_receiver.try_recv() {
                Ok(RenderMessage::WindowRecreated { window, size }) => {
                    return Some(NativeRendererBootstrap::Window { window, size });
                }
                Ok(RenderMessage::WindowRecreatedWithInit {
                    window,
                    size,
                    render_init,
                }) => {
                    return Some(NativeRendererBootstrap::Init {
                        window,
                        render_init,
                        size,
                    });
                }
                Ok(RenderMessage::Shutdown) => return None,
                Ok(message) => pending_messages.push_back(message),
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => return None,
            }
        }

        match control_receiver.recv_timeout(Duration::from_millis(16)) {
            Ok(RenderMessage::WindowRecreated { window, size }) => {
                return Some(NativeRendererBootstrap::Window { window, size });
            }
            Ok(RenderMessage::WindowRecreatedWithInit {
                window,
                size,
                render_init,
            }) => {
                return Some(NativeRendererBootstrap::Init {
                    window,
                    render_init,
                    size,
                });
            }
            Ok(RenderMessage::Shutdown) => return None,
            Ok(message) => pending_messages.push_back(message),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return None,
        }

        while let Ok(message) = render_receiver.try_recv() {
            pending_messages.push_back(message);
        }
    }
    None
}

fn release_pending_uploads(
    tuning: &Arc<RuntimeTuning>,
    asset_backlog: &mut PendingAssetUploadQueue,
    immediate_backlog: &mut PendingAssetUploadQueue,
) {
    asset_backlog.drain_release(tuning);
    immediate_backlog.drain_release(tuning);
}

#[cfg(not(target_arch = "wasm32"))]
fn create_renderer_for_window(
    window: &Arc<Window>,
    size: PhysicalSize<u32>,
    stream_request_sender: &Sender<AssetStreamingRequest>,
    stats: &Arc<RendererStats>,
    tuning: &Arc<RuntimeTuning>,
    config: RenderRuntimeConfig,
) -> Result<crate::graphics::renderer::GraphRenderer, String> {
    let render_init = create_native_render_init(window)?;
    create_renderer_from_init(
        render_init,
        size,
        stream_request_sender,
        stats,
        tuning,
        config,
    )
}

#[cfg(not(target_arch = "wasm32"))]
pub fn create_native_render_bootstrap_message(
    window: Arc<Window>,
    size: PhysicalSize<u32>,
) -> Result<RenderMessage, String> {
    let render_init = create_native_render_init(&window)?;
    Ok(RenderMessage::WindowRecreatedWithInit {
        window,
        size: PhysicalSize::new(size.width.max(1), size.height.max(1)),
        render_init,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn create_native_render_init(window: &Arc<Window>) -> Result<NativeRenderInit, String> {
    let instance = make_wgpu_instance();
    let surface = create_surface(&instance, window)?;
    Ok(NativeRenderInit { instance, surface })
}

#[cfg(not(target_arch = "wasm32"))]
fn create_renderer_from_init(
    render_init: NativeRenderInit,
    size: PhysicalSize<u32>,
    stream_request_sender: &Sender<AssetStreamingRequest>,
    stats: &Arc<RendererStats>,
    tuning: &Arc<RuntimeTuning>,
    config: RenderRuntimeConfig,
) -> Result<crate::graphics::renderer::GraphRenderer, String> {
    let target_tickrate = tuning.load_target_tickrate();
    pollster::block_on(async {
        initialize_renderer(
            render_init.instance,
            render_init.surface,
            size,
            target_tickrate,
            stream_request_sender.clone(),
            Arc::clone(stats),
            config.wgpu_experimental_features,
            config.wgpu_backend,
            config.binding_backend,
        )
        .await
        .map_err(|err| err.to_string())
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn make_wgpu_instance() -> wgpu::Instance {
    wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        backend_options: wgpu::BackendOptions {
            dx12: wgpu::Dx12BackendOptions {
                shader_compiler: wgpu::Dx12Compiler::StaticDxc,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    })
}

#[cfg(target_arch = "wasm32")]
async fn make_wgpu_instance(backend_choice: WgpuBackend) -> wgpu::Instance {
    match backend_choice {
        WgpuBackend::Gl => wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        }),
        _ => {
            let desc = wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
                ..Default::default()
            };
            wgpu::util::new_instance_with_webgpu_detection(&desc).await
        }
    }
}

fn create_surface(
    instance: &wgpu::Instance,
    window: &Arc<Window>,
) -> Result<wgpu::Surface<'static>, String> {
    #[cfg(target_os = "windows")]
    {
        use raw_window_handle::{RawDisplayHandle, WindowsDisplayHandle};
        use winit::platform::windows::WindowExtWindows;

        let raw_window_handle = unsafe {
            window
                .window_handle_any_thread()
                .map_err(|err| err.to_string())?
                .as_raw()
        };
        let raw_display_handle = RawDisplayHandle::Windows(WindowsDisplayHandle::new());
        unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle,
                    raw_window_handle,
                })
                .map_err(|err| err.to_string())
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        instance
            .create_surface(window.clone())
            .map_err(|err| err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use super::{PendingAssetUploadQueue, enqueue_asset_upload};
    use crate::graphics::common::renderer::{Aabb, MeshLodPayload, RenderMessage, Vertex};
    use crate::runtime::RuntimeTuning;
    use glam::Vec3;

    fn mesh_message(id: usize, vertex_count: usize) -> RenderMessage {
        RenderMessage::CreateMesh {
            id,
            total_lods: 1,
            lods: vec![MeshLodPayload {
                lod_index: 0,
                vertices: vec![
                    Vertex::new(
                        [0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0],
                        [1.0, 0.0, 0.0, 1.0],
                    );
                    vertex_count.max(1)
                ]
                .into(),
                indices: Vec::<u32>::new().into(),
                meshlets: Default::default(),
            }],
            bounds: Aabb {
                min: Vec3::ZERO,
                max: Vec3::ONE,
            },
            pinned: false,
        }
    }

    #[test]
    fn enqueue_asset_upload_replaces_superseded_mesh_payloads() {
        let mut backlog = PendingAssetUploadQueue::default();
        let mut immediate = PendingAssetUploadQueue::default();

        let released = enqueue_asset_upload(
            &mut backlog,
            &mut immediate,
            mesh_message(9, 4),
            true,
            8,
            usize::MAX,
        );
        assert_eq!(released, 0);
        assert_eq!(backlog.len(), 1);

        let first_bytes =
            crate::graphics::common::renderer::render_message_payload_bytes(&mesh_message(9, 4));
        let second_bytes =
            crate::graphics::common::renderer::render_message_payload_bytes(&mesh_message(9, 8));
        let released = enqueue_asset_upload(
            &mut backlog,
            &mut immediate,
            mesh_message(9, 8),
            true,
            8,
            usize::MAX,
        );
        assert_eq!(released, first_bytes);
        assert_eq!(backlog.len(), 1);
        assert_eq!(backlog.total_bytes(), second_bytes);

        let (_, queued_bytes) = backlog.pop_front().expect("latest mesh upload queued");
        assert_eq!(queued_bytes, second_bytes);
    }

    #[test]
    fn enqueue_asset_upload_coalesces_across_backlog_and_immediate_queues() {
        let mut backlog = PendingAssetUploadQueue::default();
        let mut immediate = PendingAssetUploadQueue::default();

        let first = mesh_message(5, 4);
        let first_bytes = crate::graphics::common::renderer::render_message_payload_bytes(&first);
        let released =
            enqueue_asset_upload(&mut backlog, &mut immediate, first, true, 1, usize::MAX);
        assert_eq!(released, 0);
        assert_eq!(backlog.len(), 1);

        let second = mesh_message(5, 6);
        let second_bytes = crate::graphics::common::renderer::render_message_payload_bytes(&second);
        let released = enqueue_asset_upload(&mut backlog, &mut immediate, second, false, 0, 0);
        assert_eq!(released, first_bytes);
        assert_eq!(backlog.len(), 0);
        assert_eq!(immediate.len(), 1);

        let (_, queued_bytes) = immediate
            .pop_front()
            .expect("latest immediate upload queued");
        assert_eq!(queued_bytes, second_bytes);
    }

    #[test]
    fn asset_message_collection_budget_scales_with_uploads_but_stays_bounded() {
        let tuning = Arc::new(RuntimeTuning::default());

        tuning.asset_uploads_per_frame.store(1, Ordering::Relaxed);
        assert_eq!(super::asset_message_collection_budget(&tuning), 32);

        tuning.asset_uploads_per_frame.store(4, Ordering::Relaxed);
        assert_eq!(super::asset_message_collection_budget(&tuning), 128);

        tuning.asset_uploads_per_frame.store(64, Ordering::Relaxed);
        assert_eq!(super::asset_message_collection_budget(&tuning), 256);
    }

    #[test]
    fn interactive_asset_work_budget_caps_message_and_upload_work() {
        assert_eq!(super::asset_work_budgets_for_frame(128, 8, false), (128, 8));
        assert_eq!(super::asset_work_budgets_for_frame(128, 8, true), (32, 2));
        assert_eq!(super::asset_work_budgets_for_frame(16, 1, true), (16, 1));
    }
}
