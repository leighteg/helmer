use glam::Vec3;
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use std::cell::UnsafeCell;
use std::path::Path;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
use crossbeam_channel::{Receiver, Sender};
#[cfg(not(target_arch = "wasm32"))]
use parking_lot::Mutex;

#[cfg(not(target_arch = "wasm32"))]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(not(target_arch = "wasm32"))]
use cpal::{Sample, SampleFormat, StreamConfig};

#[cfg(not(target_arch = "wasm32"))]
use symphonia::core::audio::SampleBuffer;
#[cfg(not(target_arch = "wasm32"))]
use symphonia::core::codecs::{CODEC_TYPE_NULL, Decoder, DecoderOptions};
#[cfg(not(target_arch = "wasm32"))]
use symphonia::core::formats::{FormatOptions, FormatReader, Packet};
#[cfg(not(target_arch = "wasm32"))]
use symphonia::core::io::MediaSourceStream;
#[cfg(not(target_arch = "wasm32"))]
use symphonia::core::meta::MetadataOptions;
#[cfg(not(target_arch = "wasm32"))]
use symphonia::core::probe::Hint;
#[cfg(not(target_arch = "wasm32"))]
use symphonia::core::units::TimeBase;
#[cfg(not(target_arch = "wasm32"))]
use symphonia::default::{get_codecs, get_probe};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioBus {
    Master,
    Music,
    Sfx,
    Ui,
    Ambience,
    World,
    Custom(u32),
}

impl AudioBus {
    pub const DEFAULTS: [AudioBus; 6] = [
        AudioBus::Master,
        AudioBus::Music,
        AudioBus::Sfx,
        AudioBus::Ui,
        AudioBus::Ambience,
        AudioBus::World,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioPlaybackState {
    Playing,
    Paused,
    Stopped,
}

impl Default for AudioPlaybackState {
    fn default() -> Self {
        AudioPlaybackState::Playing
    }
}

const DEFAULT_SAMPLE_RATE: u32 = 48_000;
const DEFAULT_OUTPUT_CHANNELS: u16 = 2;
const SPEED_OF_SOUND: f32 = 343.0;
const DEFAULT_HEAD_WIDTH: f32 = 0.18;
const DEFAULT_STREAM_BUFFER_FRAMES: usize = 8_192;
const DEFAULT_STREAM_CHUNK_FRAMES: usize = 2_048;
const OUTPUT_BUFFER_ALIGNMENT: u32 = 256;
const MAX_UNCOMPRESSED_STREAM_BYTES: usize = 64 * 1024 * 1024;

fn align_output_buffer_frames(frames: u32) -> u32 {
    if frames == 0 {
        return 0;
    }
    let alignment = OUTPUT_BUFFER_ALIGNMENT.max(1);
    ((frames + alignment - 1) / alignment) * alignment
}

#[cfg(not(target_arch = "wasm32"))]
fn sample_format_rank(format: SampleFormat) -> u8 {
    match format {
        SampleFormat::F32 => 0,
        SampleFormat::F64 => 1,
        SampleFormat::I16 => 2,
        SampleFormat::I32 => 3,
        SampleFormat::I8 => 4,
        SampleFormat::I64 => 5,
        SampleFormat::U16 => 6,
        SampleFormat::U32 => 7,
        SampleFormat::U8 => 8,
        SampleFormat::U64 => 9,
        _ => 10,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioLoadMode {
    Static,
    Streaming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    Wav,
    Aiff,
    Ogg,
    Flac,
    Mp3,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum AudioClipData {
    /// Interleaved PCM samples
    Pcm {
        channels: u16,
        sample_rate: u32,
        frames: Arc<Vec<f32>>,
    },
    Encoded {
        format: AudioFormat,
        bytes: Arc<Vec<u8>>,
    },
}

#[derive(Debug, Clone)]
pub struct AudioClip {
    pub name: String,
    pub data: AudioClipData,
    pub duration_seconds: Option<f32>,
    pub size_bytes: usize,
    pub channels: u16,
    pub sample_rate: u32,
    pub load_mode: AudioLoadMode,
}

impl AudioClip {
    pub fn from_bytes(name: String, bytes: &[u8]) -> Result<Self, String> {
        Self::from_bytes_with_mode(name, bytes, AudioLoadMode::Static)
    }

    pub fn from_bytes_with_mode(
        name: String,
        bytes: &[u8],
        load_mode: AudioLoadMode,
    ) -> Result<Self, String> {
        let format = detect_format(&name, bytes);
        let header_info = probe_audio_header_info(&name, bytes);
        let force_static_pcm = load_mode == AudioLoadMode::Streaming
            && matches!(format, AudioFormat::Wav | AudioFormat::Aiff)
            && bytes.len() <= MAX_UNCOMPRESSED_STREAM_BYTES;
        let resolved_mode = if force_static_pcm {
            AudioLoadMode::Static
        } else {
            load_mode
        };

        if resolved_mode == AudioLoadMode::Static {
            if format == AudioFormat::Wav {
                if let Ok(pcm) = parse_wav(bytes) {
                    let frame_count = pcm.frames.len() / pcm.channels as usize;
                    let duration_seconds = if pcm.sample_rate > 0 {
                        Some(frame_count as f32 / pcm.sample_rate as f32)
                    } else {
                        None
                    };
                    let size_bytes = pcm.frames.len() * std::mem::size_of::<f32>();
                    return Ok(AudioClip {
                        name,
                        data: AudioClipData::Pcm {
                            channels: pcm.channels,
                            sample_rate: pcm.sample_rate,
                            frames: Arc::new(pcm.frames),
                        },
                        duration_seconds,
                        size_bytes,
                        channels: pcm.channels,
                        sample_rate: pcm.sample_rate,
                        load_mode: resolved_mode,
                    });
                }
            }
            if format == AudioFormat::Aiff {
                if let Ok(pcm) = parse_aiff(bytes) {
                    let frame_count = pcm.frames.len() / pcm.channels as usize;
                    let duration_seconds = if pcm.sample_rate > 0 {
                        Some(frame_count as f32 / pcm.sample_rate as f32)
                    } else {
                        None
                    };
                    let size_bytes = pcm.frames.len() * std::mem::size_of::<f32>();
                    return Ok(AudioClip {
                        name,
                        data: AudioClipData::Pcm {
                            channels: pcm.channels,
                            sample_rate: pcm.sample_rate,
                            frames: Arc::new(pcm.frames),
                        },
                        duration_seconds,
                        size_bytes,
                        channels: pcm.channels,
                        sample_rate: pcm.sample_rate,
                        load_mode: resolved_mode,
                    });
                }
            }

            #[cfg(not(target_arch = "wasm32"))]
            if let Ok(mut pcm) = decode_symphonia(&name, bytes) {
                if let Some((_, header_rate, _)) = header_info {
                    let preferred = prefer_sample_rate(Some(header_rate), pcm.sample_rate);
                    if preferred > 0 {
                        pcm.sample_rate = preferred;
                    }
                }
                let frame_count = pcm.frames.len() / pcm.channels as usize;
                let duration_seconds = if pcm.sample_rate > 0 {
                    Some(frame_count as f32 / pcm.sample_rate as f32)
                } else {
                    None
                };
                let size_bytes = pcm.frames.len() * std::mem::size_of::<f32>();
                return Ok(AudioClip {
                    name,
                    data: AudioClipData::Pcm {
                        channels: pcm.channels,
                        sample_rate: pcm.sample_rate,
                        frames: Arc::new(pcm.frames),
                    },
                    duration_seconds,
                    size_bytes,
                    channels: pcm.channels,
                    sample_rate: pcm.sample_rate,
                    load_mode: resolved_mode,
                });
            }
        }

        let probe_info = probe_audio_info(&name, bytes);
        let info = Some(reconcile_audio_info(header_info, probe_info));
        let (channels, sample_rate, duration_seconds) = info.unwrap_or((0, 0, None));
        Ok(AudioClip {
            name,
            data: AudioClipData::Encoded {
                format,
                bytes: Arc::new(bytes.to_vec()),
            },
            duration_seconds,
            size_bytes: bytes.len(),
            channels,
            sample_rate,
            load_mode: resolved_mode,
        })
    }

    pub fn from_path(path: &Path) -> Result<Self, String> {
        Self::from_path_with_mode(path, AudioLoadMode::Static)
    }

    pub fn from_path_with_mode(path: &Path, load_mode: AudioLoadMode) -> Result<Self, String> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = path;
            return Err("Audio loading from path not supported on wasm".to_string());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let bytes = std::fs::read(path).map_err(|err| err.to_string())?;
            let name = path.to_string_lossy().to_string();
            Self::from_bytes_with_mode(name, &bytes, load_mode)
        }
    }

    pub fn is_decoded(&self) -> bool {
        matches!(self.data, AudioClipData::Pcm { .. })
    }

    pub fn frame_count(&self) -> Option<usize> {
        match &self.data {
            AudioClipData::Pcm {
                channels, frames, ..
            } => Some(frames.len() / *channels as usize),
            _ => None,
        }
    }

    pub fn sample_rate(&self) -> Option<u32> {
        if self.sample_rate > 0 {
            Some(self.sample_rate)
        } else {
            None
        }
    }

    pub fn channels(&self) -> Option<u16> {
        if self.channels > 0 {
            Some(self.channels)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AudioEmitterSettings {
    pub bus: AudioBus,
    pub volume: f32,
    pub pitch: f32,
    pub looping: bool,
    pub spatial: bool,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: f32,
    pub spatial_blend: f32,
    pub playback_state: AudioPlaybackState,
    pub play_on_spawn: bool,
    pub scene_id: Option<u64>,
}

impl Default for AudioEmitterSettings {
    fn default() -> Self {
        Self {
            bus: AudioBus::Sfx,
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            spatial: true,
            min_distance: 1.0,
            max_distance: 50.0,
            rolloff: 1.0,
            spatial_blend: 1.0,
            playback_state: AudioPlaybackState::Playing,
            play_on_spawn: true,
            scene_id: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AudioListenerSettings {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
}

impl Default for AudioListenerSettings {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: Vec3::Z,
            up: Vec3::Y,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioEmitterSnapshot {
    pub entity_id: u64,
    pub clip: Option<Arc<AudioClip>>,
    pub settings: AudioEmitterSettings,
    pub position: Vec3,
}

#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub listener: Option<AudioListenerSettings>,
    pub emitters: Vec<AudioEmitterSnapshot>,
}

#[derive(Debug, Clone)]
pub enum AudioCommand {
    Frame(AudioFrame),
    SetEnabled(bool),
    SetBusVolume(AudioBus, f32),
    RemoveBus(AudioBus),
    SetSceneVolume(u64, f32),
    SetHeadWidth(f32),
    SetSpeedOfSound(f32),
    SetStreamingConfig {
        buffer_frames: usize,
        chunk_frames: usize,
    },
    ClearEmitters,
}

#[derive(Clone, Copy, Debug)]
struct AudioResampler {
    pos: f64,
    source_rate: u32,
    engine_rate: u32,
    pitch: f32,
    step: f64,
}

impl AudioResampler {
    fn new(source_rate: u32, engine_rate: u32, pitch: f32) -> Self {
        let mut resampler = Self {
            pos: 0.0,
            source_rate: source_rate.max(1),
            engine_rate: engine_rate.max(1),
            pitch,
            step: 1.0,
        };
        resampler.recompute_step();
        resampler
    }

    fn reset(&mut self) {
        self.pos = 0.0;
    }

    fn cursor(&self) -> f64 {
        self.pos
    }

    fn advance(&mut self) {
        self.pos += self.step;
    }

    fn set_engine_rate(&mut self, engine_rate: u32) {
        let engine_rate = engine_rate.max(1);
        if self.engine_rate == engine_rate {
            return;
        }
        self.engine_rate = engine_rate;
        self.recompute_step();
    }

    fn set_source_rate(&mut self, source_rate: u32) {
        let source_rate = source_rate.max(1);
        if self.source_rate == source_rate {
            return;
        }
        // Preserve the absolute time by scaling the cursor to the new rate
        self.pos = self.pos * (source_rate as f64 / self.source_rate as f64);
        self.source_rate = source_rate;
        self.recompute_step();
    }

    fn set_pitch(&mut self, pitch: f32) {
        if (self.pitch - pitch).abs() <= f32::EPSILON {
            return;
        }
        self.pitch = pitch;
        self.recompute_step();
    }

    fn recompute_step(&mut self) {
        let pitch = self.pitch.max(0.01) as f64;
        self.step = (self.source_rate as f64 / self.engine_rate as f64) * pitch;
    }
}

struct AudioEmitterState {
    clip: Arc<AudioClip>,
    resampler: AudioResampler,
    last_frame_count: usize,
    settings: AudioEmitterSettings,
    position: Vec3,
    stream_sample_rate: Option<u32>,
    delay: BinauralDelay,
    stream: Option<AudioStreamState>,
    alive: bool,
}

struct BinauralDelay {
    left: Vec<f32>,
    right: Vec<f32>,
    index: usize,
    left_delay: usize,
    right_delay: usize,
}

impl BinauralDelay {
    fn new(max_delay: usize) -> Self {
        let size = max_delay.max(1) + 1;
        Self {
            left: vec![0.0; size],
            right: vec![0.0; size],
            index: 0,
            left_delay: 0,
            right_delay: 0,
        }
    }

    fn resize(&mut self, max_delay: usize) {
        let size = max_delay.max(1) + 1;
        if self.left.len() != size {
            self.left = vec![0.0; size];
            self.right = vec![0.0; size];
            self.index = 0;
            self.left_delay = 0;
            self.right_delay = 0;
        }
    }

    fn write(&mut self, sample: f32) -> (f32, f32) {
        let size = self.left.len();
        self.left[self.index] = sample;
        self.right[self.index] = sample;

        let left_index = (self.index + size - (self.left_delay % size)) % size;
        let right_index = (self.index + size - (self.right_delay % size)) % size;
        let left = self.left[left_index];
        let right = self.right[right_index];

        self.index = (self.index + 1) % size;
        (left, right)
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct SymphoniaStream {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    channels: u16,
    sample_rate: u32,
    codec_rate: Option<u32>,
    time_base: Option<TimeBase>,
}

#[cfg(not(target_arch = "wasm32"))]
impl SymphoniaStream {
    fn new(name: &str, bytes: Arc<Vec<u8>>) -> Result<Self, String> {
        let mut hint = Hint::new();
        if let Some(ext) = Path::new(name).extension().and_then(|s| s.to_str()) {
            hint.with_extension(ext);
        }

        let cursor = std::io::Cursor::new(bytes.as_ref().clone());
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
        let probed = get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| e.to_string())?;

        let mut format = probed.format;
        let (track_id, channels, sample_rate, codec_params) = {
            let track = format
                .default_track()
                .ok_or_else(|| "Audio stream missing default track".to_string())?;
            if track.codec_params.codec == CODEC_TYPE_NULL {
                return Err("Unsupported audio codec".to_string());
            }
            let channels = track
                .codec_params
                .channels
                .map(|c| c.count() as u16)
                .unwrap_or(0);
            let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
            let codec_params = track.codec_params.clone();
            (track.id, channels, sample_rate, codec_params)
        };

        let decoder = get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .map_err(|e| e.to_string())?;

        Ok(Self {
            format,
            decoder,
            track_id,
            channels,
            sample_rate,
            codec_rate: codec_params.sample_rate,
            time_base: codec_params.time_base,
        })
    }

    fn decode_next_chunk(&mut self, min_frames: usize, output: &mut Vec<f32>) -> bool {
        let mut frames_decoded = 0usize;
        loop {
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(_) => return frames_decoded > 0,
            };
            if packet.track_id() != self.track_id {
                continue;
            }
            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(_) => continue,
            };

            let spec = *decoded.spec();
            let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
            buf.copy_interleaved_ref(decoded);
            let channels = spec.channels.count().max(1);
            let decoded_frames = buf.samples().len() / channels;
            self.channels = channels as u16;
            let derived = derive_packet_sample_rate(&packet, self.time_base, decoded_frames);
            if let Some(picked) = pick_sample_rate(self.codec_rate, Some(spec.rate), derived) {
                self.sample_rate = picked;
            }

            output.extend_from_slice(buf.samples());
            frames_decoded += decoded_frames;

            if frames_decoded >= min_frames {
                return true;
            }
        }
    }
}

struct StreamResampler {
    source_rate: u32,
    output_rate: u32,
    ratio: f64,
    pos: f64,
    source_offset: u64,
    prev: Vec<f32>,
    has_prev: bool,
    channels: usize,
}

impl StreamResampler {
    fn new(source_rate: u32, output_rate: u32, channels: u16) -> Self {
        let channels = channels.max(1) as usize;
        let mut resampler = Self {
            source_rate: source_rate.max(1),
            output_rate: output_rate.max(1),
            ratio: 1.0,
            pos: 0.0,
            source_offset: 0,
            prev: vec![0.0; channels],
            has_prev: false,
            channels,
        };
        resampler.recompute_ratio();
        resampler
    }

    fn reset(&mut self) {
        self.pos = 0.0;
        self.source_offset = 0;
        self.has_prev = false;
        for sample in self.prev.iter_mut() {
            *sample = 0.0;
        }
    }

    fn set_source_rate(&mut self, source_rate: u32) {
        let source_rate = source_rate.max(1);
        if self.source_rate == source_rate {
            return;
        }
        self.source_rate = source_rate;
        self.recompute_ratio();
    }

    fn set_output_rate(&mut self, output_rate: u32) {
        let output_rate = output_rate.max(1);
        if self.output_rate == output_rate {
            return;
        }
        self.output_rate = output_rate;
        self.recompute_ratio();
    }

    fn set_channels(&mut self, channels: u16) {
        let channels = channels.max(1) as usize;
        if self.channels == channels {
            return;
        }
        self.channels = channels;
        self.prev.resize(channels, 0.0);
        self.reset();
    }

    fn recompute_ratio(&mut self) {
        self.ratio = self.source_rate as f64 / self.output_rate as f64;
        if !self.ratio.is_finite() || self.ratio <= f64::EPSILON {
            self.ratio = 1.0;
        }
    }

    fn resample_interleaved_chunk(&mut self, input: &[f32], channels: u16, output: &mut Vec<f32>) {
        if input.is_empty() {
            return;
        }
        let channels = channels.max(1) as usize;
        if channels != self.channels {
            self.set_channels(channels as u16);
        }
        let frame_count = input.len() / channels;
        if frame_count == 0 {
            return;
        }
        let chunk_start = self.source_offset;
        let chunk_end = chunk_start + frame_count as u64;

        if !self.has_prev {
            for ch in 0..channels {
                self.prev[ch] = input[ch];
            }
            self.has_prev = true;
        }

        let max_pos = (chunk_end - 1) as f64;
        let mut pos = self.pos;
        while pos < max_pos {
            let idx = pos.floor() as i64;
            let frac = (pos - idx as f64) as f32;
            if idx < chunk_start as i64 {
                let next_base = 0;
                for ch in 0..channels {
                    let a = self.prev[ch];
                    let b = input[next_base + ch];
                    output.push(a + (b - a) * frac);
                }
            } else {
                let i = (idx as u64 - chunk_start) as usize;
                let base = i * channels;
                let next_base = base + channels;
                if next_base + channels > input.len() {
                    break;
                }
                for ch in 0..channels {
                    let a = input[base + ch];
                    let b = input[next_base + ch];
                    output.push(a + (b - a) * frac);
                }
            }
            pos += self.ratio;
        }
        self.pos = pos;
        self.source_offset = chunk_end;
        let last_base = (frame_count - 1) * channels;
        for ch in 0..channels {
            self.prev[ch] = input[last_base + ch];
        }
    }
}

struct AudioStreamState {
    clip: Arc<AudioClip>,
    buffer: Vec<f32>,
    buffer_start_frame: u64,
    buffer_frames: usize,
    eof: bool,
    failed: bool,
    source_channels: u16,
    source_rate: u32,
    declared_rate: Option<u32>,
    output_rate: u32,
    resampler: StreamResampler,
    #[cfg(not(target_arch = "wasm32"))]
    decoder: Option<SymphoniaStream>,
}

impl AudioStreamState {
    fn new(clip: Arc<AudioClip>, output_rate: u32) -> Self {
        let source_channels = clip.channels;
        let declared_rate = clip.sample_rate();
        let source_rate = declared_rate.unwrap_or(clip.sample_rate);
        let output_rate = output_rate.max(1);
        let resampler = StreamResampler::new(source_rate.max(1), output_rate, source_channels);
        #[cfg(not(target_arch = "wasm32"))]
        let decoder = match &clip.data {
            AudioClipData::Encoded { bytes, .. } => {
                SymphoniaStream::new(&clip.name, bytes.clone()).ok()
            }
            _ => None,
        };

        #[cfg(not(target_arch = "wasm32"))]
        let failed = decoder.is_none() && !matches!(clip.data, AudioClipData::Pcm { .. });
        #[cfg(target_arch = "wasm32")]
        let failed = !matches!(clip.data, AudioClipData::Pcm { .. });

        Self {
            clip,
            buffer: Vec::new(),
            buffer_start_frame: 0,
            buffer_frames: 0,
            eof: false,
            failed,
            source_channels,
            source_rate,
            declared_rate,
            output_rate,
            resampler,
            #[cfg(not(target_arch = "wasm32"))]
            decoder,
        }
    }

    fn reset(&mut self) {
        *self = Self::new(self.clip.clone(), self.output_rate);
    }

    fn ensure_frame(
        &mut self,
        target_frame: u64,
        chunk_frames: usize,
        buffer_limit: usize,
    ) -> bool {
        if self.failed {
            return false;
        }
        if target_frame < self.buffer_start_frame {
            return false;
        }
        if target_frame < self.buffer_start_frame + self.buffer_frames as u64 {
            return true;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(decoder) = self.decoder.as_mut() else {
                self.failed = true;
                return false;
            };

            while target_frame >= self.buffer_start_frame + self.buffer_frames as u64 && !self.eof {
                let mut temp = Vec::new();
                let decoded = decoder.decode_next_chunk(chunk_frames, &mut temp);
                if !decoded {
                    self.eof = true;
                    break;
                }
                if decoder.channels > 0 && self.source_channels != decoder.channels {
                    self.source_channels = decoder.channels;
                    self.resampler.set_channels(self.source_channels);
                }
                if decoder.sample_rate > 0 && self.source_rate != decoder.sample_rate {
                    let candidate = decoder.sample_rate;
                    let preferred = prefer_sample_rate(self.declared_rate, candidate);
                    if preferred > 0 && self.source_rate != preferred {
                        self.source_rate = preferred;
                        self.resampler.set_source_rate(self.source_rate.max(1));
                    }
                }
                let frames = if self.source_channels > 0 {
                    temp.len() / self.source_channels as usize
                } else {
                    0
                };
                if frames == 0 {
                    continue;
                }
                let before = self.buffer.len();
                self.resampler.resample_interleaved_chunk(
                    &temp,
                    self.source_channels.max(1),
                    &mut self.buffer,
                );
                let channel_count = self.source_channels.max(1) as usize;
                let added_samples = self.buffer.len().saturating_sub(before);
                let added_frames = if channel_count > 0 {
                    added_samples / channel_count
                } else {
                    0
                };
                self.buffer_frames += added_frames;

                if buffer_limit > 0 && self.buffer_frames > buffer_limit {
                    let drop_frames = self.buffer_frames.saturating_sub(buffer_limit);
                    let drop_samples = drop_frames * self.source_channels.max(1) as usize;
                    if drop_samples > 0 && drop_samples < self.buffer.len() {
                        self.buffer.drain(0..drop_samples);
                        self.buffer_start_frame += drop_frames as u64;
                        self.buffer_frames -= drop_frames;
                    }
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = (target_frame, chunk_frames, buffer_limit);
        }

        target_frame < self.buffer_start_frame + self.buffer_frames as u64
    }

    fn sample_frame(
        &mut self,
        cursor: f64,
        chunk_frames: usize,
        buffer_limit: usize,
    ) -> Option<[f32; 2]> {
        if self.failed {
            return None;
        }
        let base = cursor.floor() as u64;
        let next = base + 1;

        if !self.ensure_frame(next, chunk_frames, buffer_limit) {
            return None;
        }

        let buffer_end = self.buffer_start_frame + self.buffer_frames as u64;
        if base < self.buffer_start_frame {
            return None;
        }
        if base >= buffer_end {
            return None;
        }

        let base_idx = (base - self.buffer_start_frame) as usize;
        let next_idx = if next < buffer_end {
            (next - self.buffer_start_frame) as usize
        } else {
            base_idx
        };

        let t = (cursor - base as f64) as f32;
        let channels = self.source_channels.max(1) as usize;
        if channels == 1 {
            let a = self.buffer.get(base_idx).copied().unwrap_or(0.0);
            let b = self.buffer.get(next_idx).copied().unwrap_or(a);
            let sample = a + (b - a) * t;
            return Some([sample, sample]);
        }

        let base_sample = base_idx * channels;
        let next_sample = next_idx * channels;
        if base_sample + 1 >= self.buffer.len() {
            return None;
        }
        let a_left = self.buffer[base_sample];
        let a_right = self.buffer[base_sample + 1];
        let b_left = self.buffer.get(next_sample).copied().unwrap_or(a_left);
        let b_right = self.buffer.get(next_sample + 1).copied().unwrap_or(a_right);
        Some([
            a_left + (b_left - a_left) * t,
            a_right + (b_right - a_right) * t,
        ])
    }

    fn sample_mono(
        &mut self,
        cursor: f64,
        chunk_frames: usize,
        buffer_limit: usize,
    ) -> Option<f32> {
        let frame = self.sample_frame(cursor, chunk_frames, buffer_limit)?;
        Some((frame[0] + frame[1]) * 0.5)
    }

    fn sample_stereo(
        &mut self,
        cursor: f64,
        chunk_frames: usize,
        buffer_limit: usize,
    ) -> Option<[f32; 2]> {
        self.sample_frame(cursor, chunk_frames, buffer_limit)
    }

    fn sample_rate(&self) -> Option<u32> {
        if self.output_rate > 0 {
            Some(self.output_rate)
        } else {
            None
        }
    }
}

pub struct AudioEngine {
    enabled: bool,
    sample_rate: u32,
    head_width: f32,
    speed_of_sound: f32,
    listener: AudioListenerSettings,
    buses: HashMap<AudioBus, f32>,
    scene_volumes: HashMap<u64, f32>,
    emitters: HashMap<u64, AudioEmitterState>,
    stream_caches: HashMap<usize, AudioStreamState>,
    stream_cache_used: HashSet<usize>,
    resample_cache: HashMap<(usize, u32), Arc<Vec<f32>>>,
    resample_cache_used: HashSet<(usize, u32)>,
    scratch: Vec<[f32; 2]>,
    finished_emitters: Vec<u64>,
    max_itd_samples: usize,
    streaming_buffer_frames: usize,
    streaming_chunk_frames: usize,
    #[cfg(not(target_arch = "wasm32"))]
    debug_rates: bool,
    #[cfg(not(target_arch = "wasm32"))]
    debug_logged: HashSet<usize>,
}

impl Default for AudioEngine {
    fn default() -> Self {
        let mut buses = HashMap::new();
        for bus in AudioBus::DEFAULTS {
            buses.insert(bus, 1.0);
        }
        let sample_rate = DEFAULT_SAMPLE_RATE;
        let max_itd_samples = ((sample_rate as f32) * (DEFAULT_HEAD_WIDTH / SPEED_OF_SOUND))
            .ceil()
            .max(1.0) as usize;
        #[cfg(not(target_arch = "wasm32"))]
        let debug_rates = std::env::var("HELMER_AUDIO_DEBUG_RATES")
            .map(|value| value != "0")
            .unwrap_or(false);
        Self {
            enabled: true,
            sample_rate,
            head_width: DEFAULT_HEAD_WIDTH,
            speed_of_sound: SPEED_OF_SOUND,
            listener: AudioListenerSettings::default(),
            buses,
            scene_volumes: HashMap::new(),
            emitters: HashMap::new(),
            stream_caches: HashMap::new(),
            stream_cache_used: HashSet::new(),
            resample_cache: HashMap::new(),
            resample_cache_used: HashSet::new(),
            scratch: Vec::new(),
            finished_emitters: Vec::new(),
            max_itd_samples,
            streaming_buffer_frames: DEFAULT_STREAM_BUFFER_FRAMES,
            streaming_chunk_frames: DEFAULT_STREAM_CHUNK_FRAMES,
            #[cfg(not(target_arch = "wasm32"))]
            debug_rates,
            #[cfg(not(target_arch = "wasm32"))]
            debug_logged: HashSet::new(),
        }
    }
}

impl AudioEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        let sample_rate = sample_rate.max(1);
        if self.sample_rate == sample_rate {
            return;
        }
        self.sample_rate = sample_rate;
        self.update_itd();
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn set_head_width(&mut self, head_width: f32) {
        let head_width = head_width.max(0.01);
        if (self.head_width - head_width).abs() < f32::EPSILON {
            return;
        }
        self.head_width = head_width;
        self.update_itd();
    }

    pub fn head_width(&self) -> f32 {
        self.head_width
    }

    pub fn set_speed_of_sound(&mut self, speed_of_sound: f32) {
        let speed_of_sound = speed_of_sound.max(1.0);
        if (self.speed_of_sound - speed_of_sound).abs() < f32::EPSILON {
            return;
        }
        self.speed_of_sound = speed_of_sound;
        self.update_itd();
    }

    pub fn speed_of_sound(&self) -> f32 {
        self.speed_of_sound
    }

    pub fn set_streaming_config(&mut self, buffer_frames: usize, chunk_frames: usize) {
        self.streaming_buffer_frames = buffer_frames.max(256);
        self.streaming_chunk_frames = chunk_frames.max(256);
    }

    pub fn streaming_config(&self) -> (usize, usize) {
        (self.streaming_buffer_frames, self.streaming_chunk_frames)
    }

    fn update_itd(&mut self) {
        let max_itd_samples = ((self.sample_rate as f32) * (self.head_width / self.speed_of_sound))
            .ceil()
            .max(1.0) as usize;
        self.max_itd_samples = max_itd_samples;
    }

    pub fn set_bus_volume(&mut self, bus: AudioBus, volume: f32) {
        self.buses.insert(bus, volume.max(0.0));
    }

    pub fn bus_volume(&self, bus: AudioBus) -> f32 {
        self.buses.get(&bus).copied().unwrap_or(1.0)
    }

    pub fn remove_bus(&mut self, bus: AudioBus) {
        if AudioBus::DEFAULTS.contains(&bus) {
            return;
        }
        self.buses.remove(&bus);
    }

    pub fn set_scene_volume(&mut self, scene_id: u64, volume: f32) {
        self.scene_volumes.insert(scene_id, volume.max(0.0));
    }

    pub fn scene_volume(&self, scene_id: u64) -> f32 {
        self.scene_volumes.get(&scene_id).copied().unwrap_or(1.0)
    }

    pub fn set_listener(&mut self, listener: AudioListenerSettings) {
        self.listener = listener;
    }

    pub fn begin_frame(&mut self) {
        for emitter in self.emitters.values_mut() {
            emitter.alive = false;
        }
    }

    pub fn sync_emitter(
        &mut self,
        entity_id: u64,
        clip: Arc<AudioClip>,
        settings: AudioEmitterSettings,
        position: Vec3,
    ) {
        if !self.enabled {
            return;
        }

        let frame_count = clip.frame_count().unwrap_or(0);
        let clip_rate = clip_pcm_rate(&clip)
            .or_else(|| clip.sample_rate())
            .unwrap_or(self.sample_rate);
        let entry = self
            .emitters
            .entry(entity_id)
            .or_insert_with(|| AudioEmitterState {
                clip: clip.clone(),
                resampler: AudioResampler::new(clip_rate, self.sample_rate, settings.pitch),
                last_frame_count: frame_count,
                settings,
                position,
                stream_sample_rate: clip.sample_rate(),
                delay: BinauralDelay::new(self.max_itd_samples),
                stream: None,
                alive: true,
            });

        if !Arc::ptr_eq(&entry.clip, &clip) {
            entry.clip = clip;
            entry.resampler.reset();
            let rate = clip_pcm_rate(&entry.clip)
                .or_else(|| entry.clip.sample_rate())
                .unwrap_or(self.sample_rate);
            entry.resampler.set_source_rate(rate);
            entry.stream = None;
            entry.stream_sample_rate = entry.clip.sample_rate();
        }
        if entry.clip.is_decoded() {
            entry.stream = None;
            entry.stream_sample_rate = entry.clip.sample_rate();
            let rate = clip_pcm_rate(&entry.clip)
                .or_else(|| entry.clip.sample_rate())
                .unwrap_or(self.sample_rate);
            entry.resampler.set_source_rate(rate);
        }
        entry.last_frame_count = frame_count;
        entry.settings = settings;
        entry.position = position;
        entry.alive = true;
    }

    pub fn end_frame(&mut self) {
        let mut dead: Vec<u64> = Vec::new();
        for (id, emitter) in self.emitters.iter() {
            if !emitter.alive {
                dead.push(*id);
            }
        }
        for id in dead {
            self.emitters.remove(&id);
        }
    }

    pub fn remove_emitter(&mut self, entity_id: u64) {
        self.emitters.remove(&entity_id);
    }

    pub fn clear_emitters(&mut self) {
        self.emitters.clear();
        self.stream_caches.clear();
        self.stream_cache_used.clear();
    }

    pub fn apply_frame(
        &mut self,
        listener: Option<AudioListenerSettings>,
        emitters: &[AudioEmitterSnapshot],
    ) {
        if let Some(listener) = listener {
            self.set_listener(listener);
        }
        self.begin_frame();
        for emitter in emitters {
            if let Some(clip) = &emitter.clip {
                self.sync_emitter(
                    emitter.entity_id,
                    clip.clone(),
                    emitter.settings,
                    emitter.position,
                );
            }
        }
        self.end_frame();
    }

    pub fn apply_command(&mut self, command: AudioCommand) {
        match command {
            AudioCommand::Frame(frame) => {
                self.apply_frame(frame.listener, &frame.emitters);
            }
            AudioCommand::SetEnabled(enabled) => self.set_enabled(enabled),
            AudioCommand::SetBusVolume(bus, volume) => self.set_bus_volume(bus, volume),
            AudioCommand::RemoveBus(bus) => self.remove_bus(bus),
            AudioCommand::SetSceneVolume(scene_id, volume) => {
                self.set_scene_volume(scene_id, volume)
            }
            AudioCommand::SetHeadWidth(width) => self.set_head_width(width),
            AudioCommand::SetSpeedOfSound(speed) => self.set_speed_of_sound(speed),
            AudioCommand::SetStreamingConfig {
                buffer_frames,
                chunk_frames,
            } => self.set_streaming_config(buffer_frames, chunk_frames),
            AudioCommand::ClearEmitters => self.clear_emitters(),
        }
    }

    pub fn tick(&mut self, dt: f32) {
        if !self.enabled {
            return;
        }
        let frames = (self.sample_rate as f32 * dt).ceil().max(1.0) as usize;
        self.mix(frames);
    }

    pub fn mix_into(&mut self, data: &mut [f32], channels: usize) {
        if channels == 0 {
            return;
        }
        let frames = data.len() / channels;
        let mixed = self.mix(frames);

        if channels == 1 {
            for (idx, frame) in mixed.iter().enumerate() {
                data[idx] = (frame[0] + frame[1]) * 0.5;
            }
            return;
        }

        for (idx, frame) in mixed.iter().enumerate() {
            let base = idx * channels;
            if base + 1 >= data.len() {
                break;
            }
            data[base] = frame[0];
            data[base + 1] = frame[1];
            if channels > 2 {
                let mono = (frame[0] + frame[1]) * 0.5;
                for ch in 2..channels {
                    data[base + ch] = mono;
                }
            }
        }
    }

    pub fn mix(&mut self, frames: usize) -> &[[f32; 2]] {
        self.ensure_scratch(frames);
        for sample in self.scratch.iter_mut().take(frames) {
            *sample = [0.0, 0.0];
        }
        self.finished_emitters.clear();

        if !self.enabled {
            return &self.scratch[..frames];
        }

        self.stream_cache_used.clear();
        self.resample_cache_used.clear();

        let listener = self.listener;
        let forward = listener.forward.normalize_or_zero();
        let up = listener.up.normalize_or_zero();
        let right = forward.cross(up).normalize_or_zero();

        let buses = &self.buses;
        let scene_volumes = &self.scene_volumes;
        let master_volume = buses.get(&AudioBus::Master).copied().unwrap_or(1.0);
        let max_itd_samples = self.max_itd_samples;
        let engine_rate_u32 = self.sample_rate.max(1);
        let streaming_chunk_frames = self.streaming_chunk_frames;
        let streaming_buffer_frames = self.streaming_buffer_frames;

        let (
            emitters,
            stream_caches,
            stream_cache_used,
            resample_cache,
            resample_cache_used,
            scratch,
        ) = (
            &mut self.emitters,
            &mut self.stream_caches,
            &mut self.stream_cache_used,
            &mut self.resample_cache,
            &mut self.resample_cache_used,
            &mut self.scratch,
        );

        let mut finished: Vec<u64> = Vec::new();
        let mut ended: Vec<u64> = Vec::new();
        for (id, emitter) in emitters.iter_mut() {
            if emitter.settings.playback_state == AudioPlaybackState::Stopped {
                finished.push(*id);
                continue;
            }
            if emitter.settings.playback_state == AudioPlaybackState::Paused {
                continue;
            }
            let bus_volume = buses.get(&emitter.settings.bus).copied().unwrap_or(1.0);
            let scene_volume = emitter
                .settings
                .scene_id
                .and_then(|id| scene_volumes.get(&id).copied())
                .unwrap_or(1.0);
            let gain = emitter.settings.volume.max(0.0) * bus_volume * master_volume * scene_volume;
            if gain <= 0.0 {
                continue;
            }

            let spatial = emitter.settings.spatial;
            let spatial_blend = emitter.settings.spatial_blend.clamp(0.0, 1.0);
            let spatial_active = spatial && spatial_blend > 0.0;
            let stereo_active = !spatial || spatial_blend < 1.0;
            let (pan, attenuation, left_delay, right_delay) = if spatial {
                spatial_params(
                    listener.position,
                    emitter.position,
                    right,
                    emitter.settings.min_distance,
                    emitter.settings.max_distance,
                    emitter.settings.rolloff,
                    1.0,
                    max_itd_samples,
                )
            } else {
                (0.0, 1.0, 0, 0)
            };

            if spatial_active {
                let scaled_left_delay = ((left_delay as f32) * spatial_blend).round() as usize;
                let scaled_right_delay = ((right_delay as f32) * spatial_blend).round() as usize;
                emitter.delay.resize(max_itd_samples);
                emitter.delay.left_delay = scaled_left_delay;
                emitter.delay.right_delay = scaled_right_delay;
            }

            let (left_gain, right_gain) = if spatial { pan_gains(pan) } else { (1.0, 1.0) };
            let spatial_left = gain * attenuation * left_gain;
            let spatial_right = gain * attenuation * right_gain;
            let stereo_gain = gain * attenuation;

            let clip_rate_hint = clip_pcm_rate(&emitter.clip);
            let decoded = emitter.clip.is_decoded();
            let mut source_rate_u32 = if decoded {
                clip_rate_hint
                    .or(emitter.stream_sample_rate)
                    .unwrap_or(engine_rate_u32)
                    .max(1)
            } else {
                engine_rate_u32
            };
            let pitch = emitter.settings.pitch;
            let mut resampled: Option<Arc<Vec<f32>>> = None;
            if decoded && spatial_active && !stereo_active {
                if let Some(rate) = clip_rate_hint {
                    if rate != engine_rate_u32 {
                        if let Some(frames) = resample_clip_mono_cached(
                            &emitter.clip,
                            engine_rate_u32,
                            resample_cache,
                            resample_cache_used,
                        ) {
                            resampled = Some(frames);
                            source_rate_u32 = engine_rate_u32;
                        }
                    }
                }
            }
            if decoded {
                source_rate_u32 =
                    normalize_source_rate(engine_rate_u32, source_rate_u32, emitter.clip.channels);
                source_rate_u32 = sanitize_source_rate(engine_rate_u32, source_rate_u32);
            } else {
                source_rate_u32 = engine_rate_u32;
            }
            emitter.resampler.set_engine_rate(engine_rate_u32);
            emitter.resampler.set_pitch(pitch);
            emitter.resampler.set_source_rate(source_rate_u32);

            #[cfg(not(target_arch = "wasm32"))]
            if self.debug_rates {
                let key = clip_cache_key(&emitter.clip);
                if self.debug_logged.insert(key) {
                    let clip_rate = clip_pcm_rate(&emitter.clip);
                    let clip_meta_rate = emitter.clip.sample_rate();
                    let frame_count = emitter.clip.frame_count().unwrap_or(0);
                    let duration = emitter.clip.duration_seconds;
                    tracing::debug!(
                        target: "audio.debug_rates",
                        "audio_debug_rates: '{}' decoded={} channels={} engine_rate={} source_rate={} pitch={} clip_rate={:?} meta_rate={:?} stream_rate={:?} frames={} duration={:?}",
                        emitter.clip.name,
                        decoded,
                        emitter.clip.channels,
                        engine_rate_u32,
                        source_rate_u32,
                        pitch,
                        clip_rate,
                        clip_meta_rate,
                        emitter.stream_sample_rate,
                        frame_count,
                        duration
                    );
                }
            }

            let mut frame_index = 0usize;
            while frame_index < frames {
                let cursor = emitter.resampler.cursor();
                let mut stereo_frame: Option<[f32; 2]> = None;
                let mut mono_sample: Option<f32> = None;

                if stereo_active {
                    stereo_frame = if decoded {
                        sample_stereo_from_clip(&emitter.clip, cursor)
                    } else {
                        None
                    };

                    if stereo_frame.is_none() && !decoded {
                        if emitter.stream.is_none() {
                            let cache_key = clip_cache_key(&emitter.clip);
                            if let Some(cache) = stream_caches.get_mut(&cache_key) {
                                stereo_frame = cache.sample_stereo(
                                    cursor,
                                    streaming_chunk_frames,
                                    streaming_buffer_frames,
                                );
                                if let Some(rate) = cache.sample_rate() {
                                    emitter.stream_sample_rate = Some(rate);
                                }
                            } else {
                                let mut cache =
                                    AudioStreamState::new(emitter.clip.clone(), engine_rate_u32);
                                stereo_frame = cache.sample_stereo(
                                    cursor,
                                    streaming_chunk_frames,
                                    streaming_buffer_frames,
                                );
                                if let Some(rate) = cache.sample_rate() {
                                    emitter.stream_sample_rate = Some(rate);
                                }
                                stream_caches.insert(cache_key, cache);
                            }
                            if stereo_frame.is_some() {
                                stream_cache_used.insert(cache_key);
                            }
                        }

                        if stereo_frame.is_none() {
                            if emitter.stream.is_none() {
                                emitter.stream = Some(AudioStreamState::new(
                                    emitter.clip.clone(),
                                    engine_rate_u32,
                                ));
                            }
                            if let Some(stream) = emitter.stream.as_mut() {
                                stereo_frame = stream.sample_stereo(
                                    cursor,
                                    streaming_chunk_frames,
                                    streaming_buffer_frames,
                                );
                                if let Some(rate) = stream.sample_rate() {
                                    emitter.stream_sample_rate = Some(rate);
                                }
                            }
                        }
                    }
                }

                if spatial_active {
                    if let Some(frame) = stereo_frame {
                        mono_sample = Some((frame[0] + frame[1]) * 0.5);
                    } else {
                        mono_sample = if decoded {
                            if let Some(frames) = resampled.as_deref() {
                                sample_from_mono(frames, cursor)
                            } else {
                                sample_from_clip(&emitter.clip, cursor)
                            }
                        } else {
                            None
                        };

                        if mono_sample.is_none() && !decoded {
                            if emitter.stream.is_none() {
                                let cache_key = clip_cache_key(&emitter.clip);
                                if let Some(cache) = stream_caches.get_mut(&cache_key) {
                                    mono_sample = cache.sample_mono(
                                        cursor,
                                        streaming_chunk_frames,
                                        streaming_buffer_frames,
                                    );
                                    if let Some(rate) = cache.sample_rate() {
                                        emitter.stream_sample_rate = Some(rate);
                                    }
                                } else {
                                    let mut cache = AudioStreamState::new(
                                        emitter.clip.clone(),
                                        engine_rate_u32,
                                    );
                                    mono_sample = cache.sample_mono(
                                        cursor,
                                        streaming_chunk_frames,
                                        streaming_buffer_frames,
                                    );
                                    if let Some(rate) = cache.sample_rate() {
                                        emitter.stream_sample_rate = Some(rate);
                                    }
                                    stream_caches.insert(cache_key, cache);
                                }
                                if mono_sample.is_some() {
                                    stream_cache_used.insert(cache_key);
                                }
                            }

                            if mono_sample.is_none() {
                                if emitter.stream.is_none() {
                                    emitter.stream = Some(AudioStreamState::new(
                                        emitter.clip.clone(),
                                        engine_rate_u32,
                                    ));
                                }
                                if let Some(stream) = emitter.stream.as_mut() {
                                    mono_sample = stream.sample_mono(
                                        cursor,
                                        streaming_chunk_frames,
                                        streaming_buffer_frames,
                                    );
                                    if let Some(rate) = stream.sample_rate() {
                                        emitter.stream_sample_rate = Some(rate);
                                    }
                                }
                            }
                        }
                    }
                }

                if (stereo_active && stereo_frame.is_none())
                    || (spatial_active && mono_sample.is_none())
                {
                    if emitter.settings.looping {
                        if !emitter.clip.is_decoded() && emitter.stream.is_none() {
                            emitter.stream =
                                Some(AudioStreamState::new(emitter.clip.clone(), engine_rate_u32));
                        }
                        if let Some(stream) = emitter.stream.as_mut() {
                            stream.reset();
                        }
                        emitter.resampler.reset();
                        continue;
                    }
                    emitter.settings.playback_state = AudioPlaybackState::Stopped;
                    finished.push(*id);
                    ended.push(*id);
                    break;
                }

                let target = &mut scratch[frame_index];
                match (stereo_active, spatial_active) {
                    (true, true) => {
                        let frame = stereo_frame.unwrap_or([0.0, 0.0]);
                        let mono = mono_sample.unwrap_or(0.0);
                        let (delay_left, delay_right) = emitter.delay.write(mono);
                        let spatial_left_sample = delay_left * spatial_left;
                        let spatial_right_sample = delay_right * spatial_right;
                        let stereo_left_sample = frame[0] * stereo_gain;
                        let stereo_right_sample = frame[1] * stereo_gain;
                        let inv = 1.0 - spatial_blend;
                        target[0] += stereo_left_sample * inv + spatial_left_sample * spatial_blend;
                        target[1] +=
                            stereo_right_sample * inv + spatial_right_sample * spatial_blend;
                    }
                    (false, true) => {
                        let mono = mono_sample.unwrap_or(0.0);
                        let (delay_left, delay_right) = emitter.delay.write(mono);
                        target[0] += delay_left * spatial_left;
                        target[1] += delay_right * spatial_right;
                    }
                    (true, false) => {
                        let frame = stereo_frame.unwrap_or([0.0, 0.0]);
                        target[0] += frame[0] * stereo_gain;
                        target[1] += frame[1] * stereo_gain;
                    }
                    (false, false) => {}
                }

                emitter.resampler.advance();
                frame_index += 1;
            }
        }

        for id in finished {
            emitters.remove(&id);
        }

        stream_caches.retain(|key, _| stream_cache_used.contains(key));
        resample_cache.retain(|key, _| resample_cache_used.contains(key));

        if !ended.is_empty() {
            self.finished_emitters.extend(ended);
        }

        &scratch[..frames]
    }

    pub fn emitter_counts(&self) -> (usize, usize) {
        let total = self.emitters.len();
        let streaming = self
            .emitters
            .values()
            .filter(|emitter| !emitter.clip.is_decoded())
            .count();
        (total, streaming)
    }

    pub fn take_finished_emitters(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.finished_emitters)
    }

    fn ensure_scratch(&mut self, frames: usize) {
        if self.scratch.len() < frames {
            self.scratch.resize(frames, [0.0, 0.0]);
        }
    }
}

fn pan_gains(pan: f32) -> (f32, f32) {
    let pan = pan.clamp(-1.0, 1.0);
    let angle = (pan + 1.0) * 0.25 * std::f32::consts::PI;
    (angle.cos(), angle.sin())
}

fn clip_pcm_rate(clip: &AudioClip) -> Option<u32> {
    match &clip.data {
        AudioClipData::Pcm { sample_rate, .. } if *sample_rate > 0 => Some(*sample_rate),
        _ => None,
    }
}

fn sanitize_source_rate(engine_rate: u32, source_rate: u32) -> u32 {
    let engine_rate = engine_rate.max(1);
    let source_rate = source_rate.max(1);
    if source_rate < 8_000 || source_rate > 384_000 {
        return engine_rate;
    }
    let min = (engine_rate / 4).max(8_000);
    let max = (engine_rate.saturating_mul(4)).min(192_000);
    source_rate.clamp(min, max)
}

fn is_common_rate(rate: u32) -> bool {
    let common_rates: [u32; 12] = [
        8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000,
        192_000,
    ];
    common_rates
        .iter()
        .any(|common| (*common as i32 - rate as i32).abs() <= 25)
}

fn normalize_source_rate(engine_rate: u32, source_rate: u32, channels: u16) -> u32 {
    let engine_rate = engine_rate.max(1);
    let source_rate = source_rate.max(1);
    if is_common_rate(source_rate) {
        return source_rate;
    }

    let channels = channels.max(1) as u32;
    let mut candidates: Vec<u32> = Vec::new();
    let factors = [
        2u32,
        3,
        4,
        6,
        8,
        channels,
        channels.saturating_mul(2),
        channels.saturating_mul(4),
    ];
    for factor in factors {
        if factor <= 1 {
            continue;
        }
        if source_rate % factor != 0 {
            continue;
        }
        let candidate = source_rate / factor;
        if candidate >= 8_000 && candidate <= 192_000 && is_common_rate(candidate) {
            candidates.push(candidate);
        }
    }
    if candidates.is_empty() {
        return source_rate;
    }
    candidates.sort_by_key(|rate| (*rate as i64 - engine_rate as i64).abs());
    candidates[0]
}

fn is_plausible_rate(rate: u32) -> bool {
    (8_000..=192_000).contains(&rate)
}

fn rate_within_tolerance(a: u32, b: u32, tolerance: f64) -> bool {
    if a == 0 || b == 0 {
        return false;
    }
    let a = a as f64;
    let b = b as f64;
    let diff = (a - b).abs();
    diff / a.max(b) <= tolerance
}

fn prefer_sample_rate(declared: Option<u32>, decoded: u32) -> u32 {
    if is_plausible_rate(decoded) {
        if let Some(declared) = declared {
            if is_plausible_rate(declared) && rate_within_tolerance(declared, decoded, 0.02) {
                return declared;
            }
        }
        return decoded;
    }

    declared
        .filter(|rate| is_plausible_rate(*rate))
        .unwrap_or(decoded)
}

fn reconcile_audio_info(
    header: Option<(u16, u32, Option<f32>)>,
    probe: Option<(u16, u32, Option<f32>)>,
) -> (u16, u32, Option<f32>) {
    match (header, probe) {
        (Some(h), Some(p)) => {
            let channels = if (1..=8).contains(&h.0) { h.0 } else { p.0 };
            let sample_rate = prefer_sample_rate(Some(h.1), p.1);
            let duration_seconds = h.2.or(p.2);
            (channels, sample_rate, duration_seconds)
        }
        (Some(h), None) => h,
        (None, Some(p)) => p,
        (None, None) => (0, 0, None),
    }
}

fn pick_sample_rate(
    codec_rate: Option<u32>,
    spec_rate: Option<u32>,
    derived_rate: Option<u32>,
) -> Option<u32> {
    if let Some(rate) = spec_rate.filter(|rate| is_plausible_rate(*rate)) {
        return Some(rate);
    }

    let mut candidates: Vec<u32> = Vec::new();
    for rate in [derived_rate, codec_rate].into_iter().flatten() {
        if is_plausible_rate(rate) {
            candidates.push(rate);
        }
    }
    if candidates.is_empty() {
        return None;
    }

    let common: Vec<u32> = candidates
        .iter()
        .copied()
        .filter(|rate| is_common_rate(*rate))
        .collect();
    let pool = if common.is_empty() {
        &candidates
    } else {
        &common
    };

    for rate in [derived_rate, codec_rate].into_iter().flatten() {
        if pool.contains(&rate) {
            return Some(rate);
        }
    }

    pool.first().copied()
}

#[cfg(not(target_arch = "wasm32"))]
fn time_base_to_rate(time_base: TimeBase) -> Option<u32> {
    if time_base.numer == 0 || time_base.denom == 0 {
        return None;
    }
    let rate = (time_base.denom as f64 / time_base.numer as f64).round() as u32;
    if is_plausible_rate(rate) {
        Some(rate)
    } else {
        None
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn derive_packet_sample_rate(
    packet: &Packet,
    time_base: Option<TimeBase>,
    frames: usize,
) -> Option<u32> {
    let time_base = time_base?;
    let dur = packet.dur();
    if dur == 0 || frames == 0 {
        return time_base_to_rate(time_base);
    }
    let seconds = (dur as f64) * (time_base.numer as f64) / (time_base.denom as f64);
    if seconds <= 0.0 {
        return time_base_to_rate(time_base);
    }
    let rate = (frames as f64 / seconds).round() as u32;
    if is_plausible_rate(rate) {
        Some(rate)
    } else {
        time_base_to_rate(time_base)
    }
}

fn spatial_params(
    listener_pos: Vec3,
    emitter_pos: Vec3,
    right: Vec3,
    min_distance: f32,
    max_distance: f32,
    rolloff: f32,
    blend: f32,
    max_itd_samples: usize,
) -> (f32, f32, usize, usize) {
    let min_distance = min_distance.max(0.01);
    let max_distance = max_distance.max(min_distance + 0.01);
    let rolloff = rolloff.max(0.0);
    let delta = emitter_pos - listener_pos;
    let distance = delta.length().max(0.0001);
    let direction = delta / distance;

    let right_dot = direction.dot(right).clamp(-1.0, 1.0);
    let pan = right_dot * blend.clamp(0.0, 1.0);

    let attenuation = if distance <= min_distance {
        1.0
    } else if distance >= max_distance {
        0.0
    } else {
        let falloff = distance - min_distance;
        1.0 / (1.0 + rolloff * falloff)
    };

    let max_samples = max_itd_samples as f32;
    let delay = (max_samples * right_dot).round() as i32;
    let (left_delay, right_delay) = if delay >= 0 {
        (delay as usize, 0)
    } else {
        (0, (-delay) as usize)
    };

    (pan, attenuation, left_delay, right_delay)
}

fn sample_from_clip(clip: &AudioClip, cursor: f64) -> Option<f32> {
    let AudioClipData::Pcm {
        channels, frames, ..
    } = &clip.data
    else {
        return None;
    };
    let channels = *channels as usize;
    if channels == 0 {
        return None;
    }
    let frame_count = frames.len() / channels;
    if frame_count == 0 {
        return None;
    }
    let base = cursor.floor() as usize;
    if base >= frame_count {
        return None;
    }
    let next = (base + 1).min(frame_count - 1);
    let t = (cursor - base as f64) as f32;

    let mut sample = 0.0;
    for ch in 0..channels {
        let idx = base * channels + ch;
        let idx_next = next * channels + ch;
        let a = frames[idx];
        let b = frames[idx_next];
        sample += a + (b - a) * t;
    }
    sample /= channels as f32;
    Some(sample)
}

fn sample_stereo_from_clip(clip: &AudioClip, cursor: f64) -> Option<[f32; 2]> {
    let AudioClipData::Pcm {
        channels, frames, ..
    } = &clip.data
    else {
        return None;
    };
    let channels = *channels as usize;
    if channels == 0 {
        return None;
    }
    let frame_count = frames.len() / channels;
    if frame_count == 0 {
        return None;
    }
    let base = cursor.floor() as usize;
    if base >= frame_count {
        return None;
    }
    let next = (base + 1).min(frame_count - 1);
    let t = (cursor - base as f64) as f32;

    if channels == 1 {
        let a = frames[base];
        let b = frames[next];
        let sample = a + (b - a) * t;
        return Some([sample, sample]);
    }

    let base_idx = base * channels;
    let next_idx = next * channels;
    if base_idx + 1 >= frames.len() || next_idx + 1 >= frames.len() {
        return None;
    }
    let a_left = frames[base_idx];
    let b_left = frames[next_idx];
    let a_right = frames[base_idx + 1];
    let b_right = frames[next_idx + 1];
    Some([
        a_left + (b_left - a_left) * t,
        a_right + (b_right - a_right) * t,
    ])
}

fn sample_from_mono(frames: &[f32], cursor: f64) -> Option<f32> {
    if frames.is_empty() {
        return None;
    }
    let frame_count = frames.len();
    let base = cursor.floor() as usize;
    if base >= frame_count {
        return None;
    }
    let next = (base + 1).min(frame_count - 1);
    let t = (cursor - base as f64) as f32;
    let a = frames[base];
    let b = frames[next];
    Some(a + (b - a) * t)
}

fn mono_from_interleaved(frames: &[f32], channels: u16) -> Vec<f32> {
    let channels = channels.max(1) as usize;
    if channels == 1 {
        return frames.to_vec();
    }
    let frame_count = frames.len() / channels;
    let mut mono = Vec::with_capacity(frame_count);
    for frame in 0..frame_count {
        let mut sum = 0.0f32;
        let base = frame * channels;
        for ch in 0..channels {
            sum += frames[base + ch];
        }
        mono.push(sum / channels as f32);
    }
    mono
}

fn resample_mono_windowed_sinc(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if input.is_empty() || in_rate == 0 || out_rate == 0 {
        return Vec::new();
    }
    if in_rate == out_rate {
        return input.to_vec();
    }

    const TAPS: i32 = 31;
    let half = TAPS / 2;
    let taps_minus_one = (TAPS - 1) as f64;
    let ratio = in_rate as f64 / out_rate as f64;
    let out_len = ((input.len() as f64) / ratio).ceil().max(1.0) as usize;
    let nyquist = 0.5;
    let mut cutoff = (out_rate as f64 / in_rate as f64) * nyquist;
    if cutoff > nyquist {
        cutoff = nyquist;
    }
    let two_cutoff = 2.0 * cutoff;
    let pi = std::f64::consts::PI;

    let mut output = Vec::with_capacity(out_len);
    for out_idx in 0..out_len {
        let pos = out_idx as f64 * ratio;
        let base = pos.floor() as i64;
        let frac = pos - base as f64;
        let mut sum = 0.0f64;
        let mut norm = 0.0f64;

        for i in -half..=half {
            let idx = base + i as i64;
            if idx < 0 || idx >= input.len() as i64 {
                continue;
            }
            let t = (i as f64 - frac);
            let x = t * two_cutoff;
            let sinc = if x.abs() < 1.0e-8 {
                1.0
            } else {
                (pi * x).sin() / (pi * x)
            };
            let n = (i + half) as f64;
            let window = 0.54 - 0.46 * (2.0 * pi * n / taps_minus_one).cos();
            let weight = sinc * window * two_cutoff;
            sum += input[idx as usize] as f64 * weight;
            norm += weight;
        }

        if norm.abs() > 1.0e-8 {
            output.push((sum / norm) as f32);
        } else {
            output.push(0.0);
        }
    }
    output
}

fn resample_clip_mono_cached(
    clip: &Arc<AudioClip>,
    target_rate: u32,
    cache: &mut HashMap<(usize, u32), Arc<Vec<f32>>>,
    used: &mut HashSet<(usize, u32)>,
) -> Option<Arc<Vec<f32>>> {
    let AudioClipData::Pcm {
        channels,
        sample_rate,
        frames,
    } = &clip.data
    else {
        return None;
    };
    if *sample_rate == 0 || *sample_rate == target_rate {
        return None;
    }
    let channel_count = (*channels).max(1);
    let frame_count = frames.len() / channel_count as usize;
    const RESAMPLE_CACHE_MAX_SECONDS: f32 = 12.0;
    let max_frames = ((*sample_rate as f32) * RESAMPLE_CACHE_MAX_SECONDS).ceil() as usize;
    if frame_count > max_frames {
        return None;
    }
    let key = (clip_cache_key(clip), target_rate);
    if let Some(cached) = cache.get(&key) {
        used.insert(key);
        return Some(cached.clone());
    }
    let mono = mono_from_interleaved(frames, channel_count);
    let resampled = resample_mono_windowed_sinc(&mono, *sample_rate, target_rate);
    let arc = Arc::new(resampled);
    cache.insert(key, arc.clone());
    used.insert(key);
    Some(arc)
}

fn clip_cache_key(clip: &Arc<AudioClip>) -> usize {
    Arc::as_ptr(clip) as usize
}

fn detect_format(name: &str, bytes: &[u8]) -> AudioFormat {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".wav") || bytes.starts_with(b"RIFF") {
        return AudioFormat::Wav;
    }
    if lower.ends_with(".aiff")
        || lower.ends_with(".aif")
        || lower.ends_with(".aifc")
        || (bytes.len() >= 12
            && bytes.starts_with(b"FORM")
            && (&bytes[8..12] == b"AIFF" || &bytes[8..12] == b"AIFC"))
    {
        return AudioFormat::Aiff;
    }
    if lower.ends_with(".ogg") || bytes.starts_with(b"OggS") {
        return AudioFormat::Ogg;
    }
    if lower.ends_with(".flac") || bytes.starts_with(b"fLaC") {
        return AudioFormat::Flac;
    }
    if lower.ends_with(".mp3") || bytes.starts_with(b"ID3") {
        return AudioFormat::Mp3;
    }
    AudioFormat::Unknown
}

fn probe_audio_header_info(name: &str, bytes: &[u8]) -> Option<(u16, u32, Option<f32>)> {
    let format = detect_format(name, bytes);
    match format {
        AudioFormat::Wav => parse_wav_info(bytes),
        AudioFormat::Aiff => parse_aiff_info(bytes),
        AudioFormat::Ogg => parse_ogg_info(bytes),
        AudioFormat::Flac => parse_flac_info(bytes),
        AudioFormat::Mp3 => parse_mp3_info(bytes),
        AudioFormat::Unknown => None,
    }
}

fn probe_audio_info(name: &str, bytes: &[u8]) -> Option<(u16, u32, Option<f32>)> {
    let format = detect_format(name, bytes);
    if format == AudioFormat::Wav {
        return parse_wav_info(bytes);
    }
    if format == AudioFormat::Aiff {
        return parse_aiff_info(bytes);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        return probe_audio_info_symphonia(name, bytes);
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = (name, bytes);
        None
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn probe_audio_info_symphonia(name: &str, bytes: &[u8]) -> Option<(u16, u32, Option<f32>)> {
    let mut hint = Hint::new();
    if let Some(ext) = Path::new(name).extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let cursor = std::io::Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let probed = get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;
    let mut format = probed.format;
    let track = format.default_track()?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    let mut channels = codec_params.channels.map(|c| c.count() as u16).unwrap_or(0);
    let mut sample_rate = codec_params.sample_rate.unwrap_or(0);

    if codec_params.codec != CODEC_TYPE_NULL {
        if let Ok(mut decoder) = get_codecs().make(&codec_params, &DecoderOptions::default()) {
            for _ in 0..4 {
                let packet = format.next_packet().ok()?;
                if packet.track_id() != track_id {
                    continue;
                }
                if let Ok(decoded) = decoder.decode(&packet) {
                    let spec = *decoded.spec();
                    let decoded_channels = spec.channels.count() as u16;
                    if decoded_channels > 0 {
                        channels = decoded_channels;
                    }
                    let derived = derive_packet_sample_rate(
                        &packet,
                        codec_params.time_base,
                        decoded.frames(),
                    );
                    if let Some(picked) =
                        pick_sample_rate(codec_params.sample_rate, Some(spec.rate), derived)
                    {
                        sample_rate = picked;
                    }
                    break;
                }
            }
        }
    }

    let duration_seconds = None;
    Some((channels, sample_rate, duration_seconds))
}

struct PcmData {
    channels: u16,
    sample_rate: u32,
    frames: Vec<f32>,
}

fn parse_wav_info(bytes: &[u8]) -> Option<(u16, u32, Option<f32>)> {
    if bytes.len() < 12 {
        return None;
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }

    let mut offset = 12;
    let mut fmt_chunk: Option<(u16, u16, u32, u16)> = None;
    let mut data_len: Option<usize> = None;

    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().ok()?) as usize;
        offset += 8;
        if offset + size > bytes.len() {
            return None;
        }
        let chunk = &bytes[offset..offset + size];
        match id {
            b"fmt " => {
                if size < 16 {
                    return None;
                }
                let audio_format = u16::from_le_bytes(chunk[0..2].try_into().ok()?);
                let channels = u16::from_le_bytes(chunk[2..4].try_into().ok()?);
                let sample_rate = u32::from_le_bytes(chunk[4..8].try_into().ok()?);
                let bits_per_sample = u16::from_le_bytes(chunk[14..16].try_into().ok()?);
                fmt_chunk = Some((audio_format, channels, sample_rate, bits_per_sample));
            }
            b"data" => {
                data_len = Some(size);
            }
            _ => {}
        }
        offset += size;
        if size % 2 == 1 {
            offset += 1;
        }
    }

    let Some((_audio_format, channels, sample_rate, bits_per_sample)) = fmt_chunk else {
        return None;
    };
    let data_len = data_len.unwrap_or(0);
    if channels == 0 || sample_rate == 0 || bits_per_sample == 0 {
        return None;
    }
    let bytes_per_frame = (channels as usize) * (bits_per_sample as usize / 8).max(1);
    let frames = if bytes_per_frame > 0 {
        data_len / bytes_per_frame
    } else {
        0
    };
    let duration_seconds = if frames > 0 {
        Some(frames as f32 / sample_rate as f32)
    } else {
        None
    };

    Some((channels, sample_rate, duration_seconds))
}

struct AiffComm {
    channels: u16,
    frames: u32,
    sample_size: u16,
    sample_rate: u32,
    compression: Option<[u8; 4]>,
}

fn parse_ieee_extended(bytes: &[u8; 10]) -> Option<f64> {
    let sign = (bytes[0] & 0x80) != 0;
    let exponent = (((bytes[0] & 0x7F) as u16) << 8) | bytes[1] as u16;
    let hi = u32::from_be_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
    let lo = u32::from_be_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);

    if exponent == 0 && hi == 0 && lo == 0 {
        return Some(0.0);
    }
    if exponent == 0x7FFF {
        return None;
    }

    let mantissa = ((hi as u64) << 32) | lo as u64;
    let fraction = mantissa as f64 / (1u64 << 63) as f64;
    let exp = exponent as i32 - 16383;
    let mut value = fraction * 2f64.powi(exp);
    if sign {
        value = -value;
    }
    Some(value)
}

fn parse_aiff_comm(chunk: &[u8], is_aifc: bool) -> Option<AiffComm> {
    if chunk.len() < 18 {
        return None;
    }
    let channels = u16::from_be_bytes(chunk[0..2].try_into().ok()?);
    let frames = u32::from_be_bytes(chunk[2..6].try_into().ok()?);
    let sample_size = u16::from_be_bytes(chunk[6..8].try_into().ok()?);
    let mut rate_bytes = [0u8; 10];
    rate_bytes.copy_from_slice(&chunk[8..18]);
    let sample_rate = parse_ieee_extended(&rate_bytes)
        .map(|value| value.round() as u32)
        .unwrap_or(0);
    let compression = if is_aifc && chunk.len() >= 22 {
        Some([chunk[18], chunk[19], chunk[20], chunk[21]])
    } else {
        None
    };
    Some(AiffComm {
        channels,
        frames,
        sample_size,
        sample_rate,
        compression,
    })
}

fn parse_aiff_info(bytes: &[u8]) -> Option<(u16, u32, Option<f32>)> {
    if bytes.len() < 12 || !bytes.starts_with(b"FORM") {
        return None;
    }
    let form = &bytes[8..12];
    let is_aifc = match form {
        b"AIFF" => false,
        b"AIFC" => true,
        _ => return None,
    };

    let mut offset = 12;
    let mut comm: Option<AiffComm> = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_be_bytes(bytes[offset + 4..offset + 8].try_into().ok()?) as usize;
        offset += 8;
        if offset + size > bytes.len() {
            return None;
        }
        let chunk = &bytes[offset..offset + size];
        if id == b"COMM" {
            comm = parse_aiff_comm(chunk, is_aifc);
            break;
        }
        offset += size;
        if size % 2 == 1 {
            offset += 1;
        }
    }

    let comm = comm?;
    if comm.channels == 0 || comm.sample_rate == 0 {
        return None;
    }
    let duration_seconds = if comm.frames > 0 {
        Some(comm.frames as f32 / comm.sample_rate as f32)
    } else {
        None
    };
    Some((comm.channels, comm.sample_rate, duration_seconds))
}

fn parse_ogg_info(bytes: &[u8]) -> Option<(u16, u32, Option<f32>)> {
    if bytes.len() < 16 {
        return None;
    }

    if let Some(index) = find_subslice(bytes, b"OpusHead") {
        if index + 19 <= bytes.len() {
            let channels = bytes[index + 9] as u16;
            // Opus always decodes to 48 kHz. The header input rate is informational only
            return Some((channels.max(1), 48_000, None));
        }
    }

    if let Some(index) = find_subslice(bytes, b"\x01vorbis") {
        if index + 16 <= bytes.len() {
            let channels = bytes[index + 11] as u16;
            let sample_rate = u32::from_le_bytes(bytes[index + 12..index + 16].try_into().ok()?);
            if channels > 0 && sample_rate > 0 {
                return Some((channels, sample_rate, None));
            }
        }
    }

    None
}

fn parse_flac_info(bytes: &[u8]) -> Option<(u16, u32, Option<f32>)> {
    if bytes.len() < 8 || !bytes.starts_with(b"fLaC") {
        return None;
    }

    let mut offset = 4;
    while offset + 4 <= bytes.len() {
        let header = bytes[offset];
        let block_type = header & 0x7F;
        let length =
            u32::from_be_bytes([0, bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
                as usize;
        offset += 4;
        if offset + length > bytes.len() {
            return None;
        }
        if block_type == 0 {
            if length < 18 {
                return None;
            }
            let info = &bytes[offset..offset + length];
            if info.len() < 18 {
                return None;
            }
            let mut stream_info: [u8; 8] = [0; 8];
            stream_info.copy_from_slice(&info[10..18]);
            let packed = u64::from_be_bytes(stream_info);
            let sample_rate = ((packed >> 44) & 0xFFFFF) as u32;
            let channels = (((packed >> 41) & 0x7) as u16) + 1;
            if sample_rate > 0 && channels > 0 {
                return Some((channels, sample_rate, None));
            }
            return None;
        }
        offset += length;
        if header & 0x80 != 0 {
            break;
        }
    }
    None
}

fn parse_mp3_info(bytes: &[u8]) -> Option<(u16, u32, Option<f32>)> {
    if bytes.len() < 4 {
        return None;
    }
    let mut offset = 0usize;
    if bytes.len() >= 10 && &bytes[0..3] == b"ID3" {
        let size = synchsafe_size(&bytes[6..10]);
        offset = 10 + size;
        if offset >= bytes.len() {
            return None;
        }
    }

    let mut idx = offset;
    while idx + 4 <= bytes.len() {
        if bytes[idx] == 0xFF && (bytes[idx + 1] & 0xE0) == 0xE0 {
            let header = u32::from_be_bytes(bytes[idx..idx + 4].try_into().ok()?);
            let version_id = (header >> 19) & 0x3;
            let layer_id = (header >> 17) & 0x3;
            if version_id == 0x1 || layer_id == 0x0 {
                idx += 1;
                continue;
            }
            let sample_rate_idx = (header >> 10) & 0x3;
            if sample_rate_idx == 0x3 {
                idx += 1;
                continue;
            }
            let sample_rate = match version_id {
                0x0 => [11_025, 12_000, 8_000][sample_rate_idx as usize],
                0x2 => [22_050, 24_000, 16_000][sample_rate_idx as usize],
                0x3 => [44_100, 48_000, 32_000][sample_rate_idx as usize],
                _ => {
                    idx += 1;
                    continue;
                }
            };
            let channel_mode = (header >> 6) & 0x3;
            let channels = if channel_mode == 0x3 { 1 } else { 2 };
            return Some((channels, sample_rate, None));
        }
        idx += 1;
    }
    None
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn synchsafe_size(bytes: &[u8]) -> usize {
    if bytes.len() < 4 {
        return 0;
    }
    let b0 = (bytes[0] & 0x7F) as usize;
    let b1 = (bytes[1] & 0x7F) as usize;
    let b2 = (bytes[2] & 0x7F) as usize;
    let b3 = (bytes[3] & 0x7F) as usize;
    (b0 << 21) | (b1 << 14) | (b2 << 7) | b3
}

fn parse_wav(bytes: &[u8]) -> Result<PcmData, String> {
    if bytes.len() < 12 {
        return Err("WAV data too small".to_string());
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Invalid WAV header".to_string());
    }

    let mut offset = 12;
    let mut fmt_chunk: Option<(u16, u16, u32, u16)> = None;
    let mut data_chunk: Option<&[u8]> = None;

    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .map_err(|_| "Invalid WAV chunk size")?,
        ) as usize;
        offset += 8;
        if offset + size > bytes.len() {
            return Err("WAV chunk size out of bounds".to_string());
        }
        let chunk = &bytes[offset..offset + size];
        match id {
            b"fmt " => {
                if size < 16 {
                    return Err("WAV fmt chunk too small".to_string());
                }
                let audio_format = u16::from_le_bytes(chunk[0..2].try_into().unwrap());
                let channels = u16::from_le_bytes(chunk[2..4].try_into().unwrap());
                let sample_rate = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                let bits_per_sample = u16::from_le_bytes(chunk[14..16].try_into().unwrap());
                fmt_chunk = Some((audio_format, channels, sample_rate, bits_per_sample));
            }
            b"data" => {
                data_chunk = Some(chunk);
            }
            _ => {}
        }
        offset += size;
        if size % 2 == 1 {
            offset += 1;
        }
    }

    let Some((audio_format, channels, sample_rate, bits_per_sample)) = fmt_chunk else {
        return Err("Missing WAV fmt chunk".to_string());
    };
    let Some(data) = data_chunk else {
        return Err("Missing WAV data chunk".to_string());
    };
    if channels == 0 || sample_rate == 0 {
        return Err("Invalid WAV channel count or sample rate".to_string());
    }

    let frames = decode_pcm_samples(data, audio_format, bits_per_sample)?;
    Ok(PcmData {
        channels,
        sample_rate,
        frames,
    })
}

enum AiffEndian {
    Big,
    Little,
}

fn decode_aiff_samples(
    data: &[u8],
    bits_per_sample: u16,
    endian: AiffEndian,
    float_mode: bool,
) -> Result<Vec<f32>, String> {
    if bits_per_sample == 0 || bits_per_sample % 8 != 0 {
        return Err("Unsupported AIFF sample size".to_string());
    }
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    if bytes_per_sample == 0 {
        return Err("Unsupported AIFF sample size".to_string());
    }

    let mut out: Vec<f32> = Vec::with_capacity(data.len() / bytes_per_sample);
    if float_mode {
        match bits_per_sample {
            32 => {
                for chunk in data.chunks_exact(4) {
                    let sample = match endian {
                        AiffEndian::Big => {
                            f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        }
                        AiffEndian::Little => {
                            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        }
                    };
                    out.push(sample);
                }
                return Ok(out);
            }
            64 => {
                for chunk in data.chunks_exact(8) {
                    let sample = match endian {
                        AiffEndian::Big => f64::from_be_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]),
                        AiffEndian::Little => f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]),
                    };
                    out.push(sample as f32);
                }
                return Ok(out);
            }
            _ => {
                return Err("Unsupported AIFF float sample size".to_string());
            }
        }
    }

    match bits_per_sample {
        8 => {
            for &b in data.iter() {
                let sample = (b as i8) as f32 / 128.0;
                out.push(sample);
            }
        }
        16 => {
            for chunk in data.chunks_exact(2) {
                let sample = match endian {
                    AiffEndian::Big => i16::from_be_bytes([chunk[0], chunk[1]]),
                    AiffEndian::Little => i16::from_le_bytes([chunk[0], chunk[1]]),
                };
                out.push(sample as f32 / 32768.0);
            }
        }
        24 => {
            for chunk in data.chunks_exact(3) {
                let value = match endian {
                    AiffEndian::Big => {
                        ((chunk[0] as i32) << 24)
                            | ((chunk[1] as i32) << 16)
                            | ((chunk[2] as i32) << 8)
                    }
                    AiffEndian::Little => {
                        ((chunk[2] as i32) << 24)
                            | ((chunk[1] as i32) << 16)
                            | ((chunk[0] as i32) << 8)
                    }
                };
                out.push((value >> 8) as f32 / 8_388_608.0);
            }
        }
        32 => {
            for chunk in data.chunks_exact(4) {
                let sample = match endian {
                    AiffEndian::Big => i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                    AiffEndian::Little => {
                        i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                    }
                };
                out.push(sample as f32 / 2_147_483_648.0);
            }
        }
        _ => return Err("Unsupported AIFF sample size".to_string()),
    }

    Ok(out)
}

fn parse_aiff(bytes: &[u8]) -> Result<PcmData, String> {
    if bytes.len() < 12 || !bytes.starts_with(b"FORM") {
        return Err("Invalid AIFF header".to_string());
    }
    let form = &bytes[8..12];
    let is_aifc = match form {
        b"AIFF" => false,
        b"AIFC" => true,
        _ => return Err("Invalid AIFF form type".to_string()),
    };

    let mut offset = 12;
    let mut comm: Option<AiffComm> = None;
    let mut ssnd: Option<&[u8]> = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_be_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .map_err(|_| "Invalid AIFF chunk size")?,
        ) as usize;
        offset += 8;
        if offset + size > bytes.len() {
            return Err("AIFF chunk size out of bounds".to_string());
        }
        let chunk = &bytes[offset..offset + size];
        match id {
            b"COMM" => {
                comm = parse_aiff_comm(chunk, is_aifc);
            }
            b"SSND" => {
                ssnd = Some(chunk);
            }
            _ => {}
        }
        offset += size;
        if size % 2 == 1 {
            offset += 1;
        }
    }

    let Some(comm) = comm else {
        return Err("Missing AIFF COMM chunk".to_string());
    };
    let Some(ssnd) = ssnd else {
        return Err("Missing AIFF SSND chunk".to_string());
    };
    if comm.channels == 0 || comm.sample_rate == 0 {
        return Err("Invalid AIFF channel count or sample rate".to_string());
    }
    if ssnd.len() < 8 {
        return Err("AIFF SSND chunk too small".to_string());
    }

    let offset = u32::from_be_bytes(ssnd[0..4].try_into().unwrap()) as usize;
    let data = &ssnd[8..];
    if offset > data.len() {
        return Err("AIFF SSND offset out of bounds".to_string());
    }
    let sample_data = &data[offset..];

    let (endian, float_mode) = match comm.compression {
        None => (AiffEndian::Big, false),
        Some(tag) => match &tag {
            b"NONE" | b"twos" | b"TWOS" => (AiffEndian::Big, false),
            b"sowt" | b"SOWT" => (AiffEndian::Little, false),
            b"fl32" | b"FL32" => (AiffEndian::Big, true),
            b"fl64" | b"FL64" => (AiffEndian::Big, true),
            _ => {
                return Err(format!(
                    "Unsupported AIFF compression: {:?}",
                    std::str::from_utf8(&tag).unwrap_or("unknown")
                ));
            }
        },
    };

    let frames = decode_aiff_samples(sample_data, comm.sample_size, endian, float_mode)?;
    Ok(PcmData {
        channels: comm.channels,
        sample_rate: comm.sample_rate,
        frames,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn decode_symphonia(name: &str, bytes: &[u8]) -> Result<PcmData, String> {
    let mut hint = Hint::new();
    if let Some(ext) = Path::new(name).extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let cursor = std::io::Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let probed = get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| e.to_string())?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "Audio stream missing default track".to_string())?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    if codec_params.codec == CODEC_TYPE_NULL {
        return Err("Unsupported audio codec".to_string());
    }

    let mut decoder = get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| e.to_string())?;

    let mut channels = codec_params.channels.map(|c| c.count() as u16).unwrap_or(0);
    let mut sample_rate = codec_params.sample_rate.unwrap_or(0);
    let codec_rate = codec_params.sample_rate;
    let mut frames: Vec<f32> = Vec::new();

    let time_base = codec_params.time_base;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(_) => continue,
        };

        let spec = *decoded.spec();
        let decoded_channels = spec.channels.count().max(1);
        channels = decoded_channels as u16;
        let derived = derive_packet_sample_rate(&packet, time_base, decoded.frames());
        if let Some(picked) = pick_sample_rate(codec_rate, Some(spec.rate), derived) {
            sample_rate = picked;
        }

        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        frames.extend_from_slice(buf.samples());
    }

    if channels == 0 || sample_rate == 0 {
        return Err("Decoded audio missing channel/sample rate".to_string());
    }

    Ok(PcmData {
        channels,
        sample_rate,
        frames,
    })
}

fn decode_pcm_samples(
    data: &[u8],
    audio_format: u16,
    bits_per_sample: u16,
) -> Result<Vec<f32>, String> {
    match (audio_format, bits_per_sample) {
        (1, 8) => Ok(data.iter().map(|b| (*b as f32 - 128.0) / 128.0).collect()),
        (1, 16) => {
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(sample as f32 / 32768.0);
            }
            Ok(out)
        }
        (1, 24) => {
            let mut out = Vec::with_capacity(data.len() / 3);
            for chunk in data.chunks_exact(3) {
                let value = ((chunk[2] as i32) << 24)
                    | ((chunk[1] as i32) << 16)
                    | ((chunk[0] as i32) << 8);
                out.push((value >> 8) as f32 / 8_388_608.0);
            }
            Ok(out)
        }
        (1, 32) => {
            let mut out = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                let sample = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(sample as f32 / 2_147_483_648.0);
            }
            Ok(out)
        }
        (3, 32) => {
            let mut out = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(sample);
            }
            Ok(out)
        }
        _ => Err(format!(
            "Unsupported WAV format: format {} bits {}",
            audio_format, bits_per_sample
        )),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioOutputSettings {
    #[cfg(not(target_arch = "wasm32"))]
    pub host_id: Option<AudioHostId>,
    pub device_name: Option<String>,
    pub device_index: Option<usize>,
    pub sample_rate: u32,
    pub channels: u16,
    pub buffer_frames: Option<u32>,
}

impl Default for AudioOutputSettings {
    fn default() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            host_id: None,
            device_name: None,
            device_index: None,
            sample_rate: DEFAULT_SAMPLE_RATE,
            channels: DEFAULT_OUTPUT_CHANNELS,
            buffer_frames: None,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioHostId(cpal::HostId);

#[cfg(not(target_arch = "wasm32"))]
impl AudioHostId {
    pub fn from_cpal(id: cpal::HostId) -> Self {
        Self(id)
    }

    pub fn to_cpal(self) -> cpal::HostId {
        self.0
    }

    pub fn label(self) -> String {
        format!("{:?}", self.0)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn host_priority(label: &str) -> i32 {
    let label = label.to_ascii_lowercase();
    if label.contains("coreaudio") {
        60
    } else if label.contains("wasapi") {
        60
    } else if label.contains("asio") {
        50
    } else if label.contains("pipewire") {
        40
    } else if label.contains("pulse") {
        30
    } else if label.contains("alsa") {
        20
    } else if label.contains("jack") {
        10
    } else {
        0
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn preferred_host_id() -> Option<cpal::HostId> {
    let mut best: Option<cpal::HostId> = None;
    let mut best_score = i32::MIN;
    for host in cpal::available_hosts() {
        let score = host_priority(&format!("{:?}", host));
        if score > best_score {
            best_score = score;
            best = Some(host);
        }
    }
    best
}

#[derive(Debug, Clone)]
pub struct AudioOutputDevice {
    pub index: usize,
    pub name: String,
}

#[derive(Debug, Clone, Copy)]
pub struct AudioRuntimeStats {
    pub enabled: bool,
    pub sample_rate: u32,
    pub measured_sample_rate: u32,
    pub channels: u16,
    pub buffer_frames: u32,
    pub mix_time_us: u64,
    pub callback_time_us: u64,
    pub active_emitters: u32,
    pub streaming_emitters: u32,
    pub frames_mixed: u64,
}

#[allow(dead_code)]
#[derive(Default)]
pub struct AudioProfiler {
    mix_time_us: AtomicU64,
    callback_time_us: AtomicU64,
    active_emitters: AtomicU32,
    streaming_emitters: AtomicU32,
    frames_mixed: AtomicU64,
    sample_rate: AtomicU32,
    measured_sample_rate: AtomicU32,
    channels: AtomicU32,
    buffer_frames: AtomicU32,
}

#[allow(dead_code)]
impl AudioProfiler {
    fn update_config(&self, sample_rate: u32, channels: u16, buffer_frames: u32) {
        self.sample_rate.store(sample_rate, Ordering::Relaxed);
        self.measured_sample_rate
            .store(sample_rate, Ordering::Relaxed);
        self.channels.store(channels as u32, Ordering::Relaxed);
        self.buffer_frames.store(buffer_frames, Ordering::Relaxed);
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct AudioRingBuffer {
    data: UnsafeCell<Vec<f32>>,
    capacity: usize,
    read: AtomicUsize,
    write: AtomicUsize,
}

#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for AudioRingBuffer {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for AudioRingBuffer {}

#[cfg(not(target_arch = "wasm32"))]
impl AudioRingBuffer {
    fn new(samples: usize) -> Self {
        let capacity = samples.max(2);
        Self {
            data: UnsafeCell::new(vec![0.0; capacity]),
            capacity,
            read: AtomicUsize::new(0),
            write: AtomicUsize::new(0),
        }
    }

    fn available_read(&self) -> usize {
        let read = self.read.load(Ordering::Acquire);
        let write = self.write.load(Ordering::Acquire);
        if write >= read {
            write - read
        } else {
            self.capacity - (read - write)
        }
    }

    fn available_write(&self) -> usize {
        let read = self.read.load(Ordering::Acquire);
        let write = self.write.load(Ordering::Acquire);
        if write >= read {
            self.capacity - (write - read) - 1
        } else {
            (read - write).saturating_sub(1)
        }
    }

    fn push_slice(&self, input: &[f32]) -> usize {
        let writable = self.available_write().min(input.len());
        if writable == 0 {
            return 0;
        }
        let mut write = self.write.load(Ordering::Relaxed);
        let first = (self.capacity - write).min(writable);
        unsafe {
            let data = &mut *self.data.get();
            data[write..write + first].copy_from_slice(&input[..first]);
            if writable > first {
                let tail = writable - first;
                data[0..tail].copy_from_slice(&input[first..first + tail]);
            }
        }
        write = (write + writable) % self.capacity;
        self.write.store(write, Ordering::Release);
        writable
    }

    fn pop_slice(&self, output: &mut [f32]) -> usize {
        let readable = self.available_read().min(output.len());
        if readable == 0 {
            return 0;
        }
        let mut read = self.read.load(Ordering::Relaxed);
        let first = (self.capacity - read).min(readable);
        unsafe {
            let data = &*self.data.get();
            output[..first].copy_from_slice(&data[read..read + first]);
            if readable > first {
                let tail = readable - first;
                output[first..first + tail].copy_from_slice(&data[0..tail]);
            }
        }
        read = (read + readable) % self.capacity;
        self.read.store(read, Ordering::Release);
        readable
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct AudioBackendState {
    enabled: bool,
    output: AudioOutputSettings,
    command_tx: Option<Sender<AudioCommand>>,
    event_rx: Option<Receiver<AudioEvent>>,
    last_error: Option<String>,
    bus_volumes: HashMap<AudioBus, f32>,
    bus_names: HashMap<AudioBus, String>,
    next_custom_bus_id: u32,
    scene_volumes: HashMap<u64, f32>,
    head_width: f32,
    speed_of_sound: f32,
    streaming_buffer_frames: usize,
    streaming_chunk_frames: usize,
}

#[cfg(not(target_arch = "wasm32"))]
enum AudioControlCommand {
    Reconfigure {
        settings: AudioOutputSettings,
        respond_to: Sender<Result<(), String>>,
    },
    SetEnabled(bool),
    Shutdown,
}

#[cfg(not(target_arch = "wasm32"))]
enum AudioEvent {
    EmittersFinished(Vec<u64>),
}

#[cfg(not(target_arch = "wasm32"))]
struct AudioRenderHandle {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl AudioRenderHandle {
    fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct AudioBackend {
    state: Arc<Mutex<AudioBackendState>>,
    profiler: Arc<AudioProfiler>,
    control_tx: Sender<AudioControlCommand>,
    control_thread: Mutex<Option<thread::JoinHandle<()>>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl AudioBackend {
    pub fn new() -> Self {
        let profiler = Arc::new(AudioProfiler::default());
        let mut bus_volumes = HashMap::new();
        for bus in AudioBus::DEFAULTS {
            bus_volumes.insert(bus, 1.0);
        }

        let state = Arc::new(Mutex::new(AudioBackendState {
            enabled: true,
            output: AudioOutputSettings::default(),
            command_tx: None,
            event_rx: None,
            last_error: None,
            bus_volumes,
            bus_names: HashMap::new(),
            next_custom_bus_id: 1,
            scene_volumes: HashMap::new(),
            head_width: DEFAULT_HEAD_WIDTH,
            speed_of_sound: SPEED_OF_SOUND,
            streaming_buffer_frames: DEFAULT_STREAM_BUFFER_FRAMES,
            streaming_chunk_frames: DEFAULT_STREAM_CHUNK_FRAMES,
        }));

        let (control_tx, control_rx) = crossbeam_channel::unbounded();
        let thread_state = Arc::clone(&state);
        let thread_profiler = Arc::clone(&profiler);
        let handle =
            thread::spawn(move || audio_control_thread(thread_state, thread_profiler, control_rx));

        let backend = Self {
            state,
            profiler,
            control_tx,
            control_thread: Mutex::new(Some(handle)),
        };

        if backend.reconfigure(backend.output_settings()).is_err() {
            let mut base_settings = backend.output_settings();
            base_settings.device_name = None;
            base_settings.device_index = None;
            for host in backend.available_output_hosts() {
                let mut settings = base_settings.clone();
                settings.host_id = Some(host);
                if backend.reconfigure(settings).is_ok() {
                    break;
                }
            }
        }
        backend
    }

    pub fn enabled(&self) -> bool {
        self.state.lock().enabled
    }

    pub fn set_enabled(&self, enabled: bool) {
        let command_tx = {
            let mut state = self.state.lock();
            state.enabled = enabled;
            state.command_tx.clone()
        };
        if let Some(tx) = command_tx.as_ref() {
            let _ = tx.try_send(AudioCommand::SetEnabled(enabled));
        }
        let _ = self
            .control_tx
            .try_send(AudioControlCommand::SetEnabled(enabled));
    }

    pub fn send_frame(
        &self,
        listener: Option<AudioListenerSettings>,
        emitters: Vec<AudioEmitterSnapshot>,
    ) {
        let command_tx = {
            let state = self.state.lock();
            state.command_tx.clone()
        };
        let Some(tx) = command_tx.as_ref() else {
            return;
        };
        let frame = AudioFrame { listener, emitters };
        let _ = tx.try_send(AudioCommand::Frame(frame));
    }

    pub fn set_bus_volume(&self, bus: AudioBus, volume: f32) {
        let command_tx = {
            let mut state = self.state.lock();
            state.bus_volumes.insert(bus, volume.max(0.0));
            state.command_tx.clone()
        };
        if let Some(tx) = command_tx.as_ref() {
            let _ = tx.try_send(AudioCommand::SetBusVolume(bus, volume));
        }
    }

    pub fn bus_volume(&self, bus: AudioBus) -> f32 {
        self.state
            .lock()
            .bus_volumes
            .get(&bus)
            .copied()
            .unwrap_or(1.0)
    }

    pub fn bus_name(&self, bus: AudioBus) -> String {
        let state = self.state.lock();
        if let Some(name) = state.bus_names.get(&bus) {
            return name.clone();
        }
        match bus {
            AudioBus::Master => "Master".to_string(),
            AudioBus::Music => "Music".to_string(),
            AudioBus::Sfx => "Sfx".to_string(),
            AudioBus::Ui => "UI".to_string(),
            AudioBus::Ambience => "Ambience".to_string(),
            AudioBus::World => "World".to_string(),
            AudioBus::Custom(id) => format!("Bus {}", id),
        }
    }

    pub fn set_bus_name(&self, bus: AudioBus, name: String) {
        if AudioBus::DEFAULTS.contains(&bus) {
            return;
        }
        let trimmed = name.trim();
        let mut state = self.state.lock();
        if trimmed.is_empty() {
            state.bus_names.remove(&bus);
        } else {
            state.bus_names.insert(bus, trimmed.to_string());
        }
    }

    pub fn create_custom_bus(&self, name: Option<String>) -> AudioBus {
        let (bus, command_tx, volume) = {
            let mut state = self.state.lock();
            let id = state.next_custom_bus_id.max(1);
            state.next_custom_bus_id = id.saturating_add(1);
            let bus = AudioBus::Custom(id);
            let volume = {
                let entry = state.bus_volumes.entry(bus).or_insert(1.0);
                *entry
            };
            if let Some(name) = name {
                let trimmed = name.trim();
                if !trimmed.is_empty() {
                    state.bus_names.insert(bus, trimmed.to_string());
                }
            }
            (bus, state.command_tx.clone(), volume)
        };
        if let Some(tx) = command_tx.as_ref() {
            let _ = tx.try_send(AudioCommand::SetBusVolume(bus, volume));
        }
        bus
    }

    pub fn remove_bus(&self, bus: AudioBus) {
        if AudioBus::DEFAULTS.contains(&bus) {
            return;
        }
        let command_tx = {
            let mut state = self.state.lock();
            state.bus_volumes.remove(&bus);
            state.bus_names.remove(&bus);
            state.command_tx.clone()
        };
        if let Some(tx) = command_tx.as_ref() {
            let _ = tx.try_send(AudioCommand::RemoveBus(bus));
        }
    }

    pub fn bus_list(&self) -> Vec<AudioBus> {
        let state = self.state.lock();
        let mut list: Vec<AudioBus> = AudioBus::DEFAULTS.to_vec();
        let mut custom_ids: Vec<u32> = state
            .bus_volumes
            .keys()
            .chain(state.bus_names.keys())
            .filter_map(|bus| {
                if let AudioBus::Custom(id) = bus {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        custom_ids.sort_unstable();
        custom_ids.dedup();
        for id in custom_ids {
            list.push(AudioBus::Custom(id));
        }
        list
    }

    pub fn set_scene_volume(&self, scene_id: u64, volume: f32) {
        let command_tx = {
            let mut state = self.state.lock();
            state.scene_volumes.insert(scene_id, volume.max(0.0));
            state.command_tx.clone()
        };
        if let Some(tx) = command_tx.as_ref() {
            let _ = tx.try_send(AudioCommand::SetSceneVolume(scene_id, volume));
        }
    }

    pub fn scene_volume(&self, scene_id: u64) -> f32 {
        self.state
            .lock()
            .scene_volumes
            .get(&scene_id)
            .copied()
            .unwrap_or(1.0)
    }

    pub fn set_head_width(&self, width: f32) {
        let (command_tx, head_width) = {
            let mut state = self.state.lock();
            state.head_width = width.max(0.01);
            (state.command_tx.clone(), state.head_width)
        };
        if let Some(tx) = command_tx.as_ref() {
            let _ = tx.try_send(AudioCommand::SetHeadWidth(head_width));
        }
    }

    pub fn head_width(&self) -> f32 {
        self.state.lock().head_width
    }

    pub fn set_speed_of_sound(&self, speed: f32) {
        let (command_tx, speed_of_sound) = {
            let mut state = self.state.lock();
            state.speed_of_sound = speed.max(1.0);
            (state.command_tx.clone(), state.speed_of_sound)
        };
        if let Some(tx) = command_tx.as_ref() {
            let _ = tx.try_send(AudioCommand::SetSpeedOfSound(speed_of_sound));
        }
    }

    pub fn speed_of_sound(&self) -> f32 {
        self.state.lock().speed_of_sound
    }

    pub fn set_streaming_config(&self, buffer_frames: usize, chunk_frames: usize) {
        let (command_tx, buffer_frames, chunk_frames) = {
            let mut state = self.state.lock();
            state.streaming_buffer_frames = buffer_frames.max(256);
            state.streaming_chunk_frames = chunk_frames.max(256);
            (
                state.command_tx.clone(),
                state.streaming_buffer_frames,
                state.streaming_chunk_frames,
            )
        };
        if let Some(tx) = command_tx.as_ref() {
            let _ = tx.try_send(AudioCommand::SetStreamingConfig {
                buffer_frames,
                chunk_frames,
            });
        }
    }

    pub fn streaming_config(&self) -> (usize, usize) {
        let state = self.state.lock();
        (state.streaming_buffer_frames, state.streaming_chunk_frames)
    }

    pub fn clear_emitters(&self) {
        let command_tx = {
            let state = self.state.lock();
            state.command_tx.clone()
        };
        if let Some(tx) = command_tx.as_ref() {
            let _ = tx.try_send(AudioCommand::ClearEmitters);
        }
    }

    pub fn output_settings(&self) -> AudioOutputSettings {
        self.state.lock().output.clone()
    }

    pub fn last_error(&self) -> Option<String> {
        self.state.lock().last_error.clone()
    }

    pub fn drain_finished_emitters(&self) -> Vec<u64> {
        let rx = {
            let state = self.state.lock();
            state.event_rx.clone()
        };
        let Some(rx) = rx else {
            return Vec::new();
        };
        let mut finished = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event {
                AudioEvent::EmittersFinished(ids) => finished.extend(ids),
            }
        }
        finished
    }

    pub fn available_output_hosts(&self) -> Vec<AudioHostId> {
        let mut hosts = cpal::available_hosts();
        hosts.sort_by_key(|host| std::cmp::Reverse(host_priority(&format!("{:?}", host))));
        hosts.into_iter().map(AudioHostId::from_cpal).collect()
    }

    pub fn available_output_devices(&self, host_id: Option<AudioHostId>) -> Vec<AudioOutputDevice> {
        let host = if let Some(host_id) = host_id {
            cpal::host_from_id(host_id.to_cpal()).unwrap_or_else(|_| cpal::default_host())
        } else if let Some(preferred) = preferred_host_id() {
            cpal::host_from_id(preferred).unwrap_or_else(|_| cpal::default_host())
        } else {
            cpal::default_host()
        };
        match host.output_devices() {
            Ok(devices) => devices
                .enumerate()
                .filter_map(|(index, device)| {
                    device
                        .name()
                        .ok()
                        .map(|name| AudioOutputDevice { index, name })
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    pub fn stats(&self) -> AudioRuntimeStats {
        let state = self.state.lock();
        AudioRuntimeStats {
            enabled: state.enabled,
            sample_rate: self.profiler.sample_rate.load(Ordering::Relaxed),
            measured_sample_rate: self.profiler.measured_sample_rate.load(Ordering::Relaxed),
            channels: self.profiler.channels.load(Ordering::Relaxed) as u16,
            buffer_frames: self.profiler.buffer_frames.load(Ordering::Relaxed),
            mix_time_us: self.profiler.mix_time_us.load(Ordering::Relaxed),
            callback_time_us: self.profiler.callback_time_us.load(Ordering::Relaxed),
            active_emitters: self.profiler.active_emitters.load(Ordering::Relaxed),
            streaming_emitters: self.profiler.streaming_emitters.load(Ordering::Relaxed),
            frames_mixed: self.profiler.frames_mixed.load(Ordering::Relaxed),
        }
    }

    pub fn reconfigure(&self, settings: AudioOutputSettings) -> Result<(), String> {
        let (tx, rx) = crossbeam_channel::bounded(1);
        if self
            .control_tx
            .send(AudioControlCommand::Reconfigure {
                settings,
                respond_to: tx,
            })
            .is_err()
        {
            return Err("Audio control thread not available".to_string());
        }
        rx.recv()
            .unwrap_or_else(|_| Err("Audio control thread not available".to_string()))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for AudioBackend {
    fn drop(&mut self) {
        let _ = self.control_tx.try_send(AudioControlCommand::Shutdown);
        if let Some(handle) = self.control_thread.lock().take() {
            let _ = handle.join();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn audio_control_thread(
    state: Arc<Mutex<AudioBackendState>>,
    profiler: Arc<AudioProfiler>,
    control_rx: Receiver<AudioControlCommand>,
) {
    let mut stream: Option<cpal::Stream> = None;
    let mut render: Option<AudioRenderHandle> = None;
    while let Ok(cmd) = control_rx.recv() {
        match cmd {
            AudioControlCommand::Reconfigure {
                settings,
                respond_to,
            } => {
                if let Some(handle) = render.take() {
                    handle.stop();
                }
                let result = configure_stream(&state, &profiler, settings);
                match result {
                    Ok((
                        new_stream,
                        output_settings,
                        command_tx,
                        event_rx,
                        enabled,
                        render_handle,
                    )) => {
                        if enabled {
                            if let Err(err) = new_stream.play() {
                                let msg = err.to_string();
                                state.lock().last_error = Some(msg.clone());
                                render_handle.stop();
                                let _ = respond_to.send(Err(msg));
                                continue;
                            }
                        }

                        {
                            let mut state = state.lock();
                            state.output = output_settings;
                            state.command_tx = Some(command_tx);
                            state.event_rx = Some(event_rx);
                            state.last_error = None;
                        }

                        stream = Some(new_stream);
                        render = Some(render_handle);
                        let _ = respond_to.send(Ok(()));
                    }
                    Err(err) => {
                        let _ = respond_to.send(Err(err));
                    }
                }
            }
            AudioControlCommand::SetEnabled(enabled) => {
                if let Some(stream) = stream.as_ref() {
                    if enabled {
                        let _ = stream.play();
                    } else {
                        let _ = stream.pause();
                    }
                }
            }
            AudioControlCommand::Shutdown => {
                if let Some(handle) = render.take() {
                    handle.stop();
                }
                break;
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn configure_stream(
    state: &Arc<Mutex<AudioBackendState>>,
    profiler: &Arc<AudioProfiler>,
    mut settings: AudioOutputSettings,
) -> Result<
    (
        cpal::Stream,
        AudioOutputSettings,
        Sender<AudioCommand>,
        Receiver<AudioEvent>,
        bool,
        AudioRenderHandle,
    ),
    String,
> {
    let set_error = |msg: String| {
        state.lock().last_error = Some(msg.clone());
        msg
    };
    let host = if let Some(host_id) = settings.host_id {
        cpal::host_from_id(host_id.to_cpal()).map_err(|e| set_error(e.to_string()))?
    } else if let Some(preferred) = preferred_host_id() {
        cpal::host_from_id(preferred).unwrap_or_else(|_| cpal::default_host())
    } else {
        cpal::default_host()
    };
    settings.host_id = Some(AudioHostId::from_cpal(host.id()));

    let mut selected_index = settings.device_index;
    let mut selected_name = settings.device_name.clone();
    let device = if let Some(index) = settings.device_index {
        host.output_devices()
            .map_err(|e| set_error(e.to_string()))?
            .nth(index)
            .ok_or_else(|| set_error("Output device not found".to_string()))?
    } else if let Some(name) = settings.device_name.as_ref() {
        let mut found: Option<(usize, cpal::Device)> = None;
        if let Ok(devices) = host.output_devices() {
            for (idx, dev) in devices.enumerate() {
                if dev.name().ok().as_ref() == Some(name) {
                    found = Some((idx, dev));
                    break;
                }
            }
        }
        if let Some((idx, dev)) = found {
            selected_index = Some(idx);
            dev
        } else {
            return Err(set_error("Output device not found".to_string()));
        }
    } else {
        let device = host
            .default_output_device()
            .ok_or_else(|| set_error("No default output device".to_string()))?;
        let default_name = device.name().ok();
        if selected_index.is_none() {
            if let Some(name) = default_name.as_ref() {
                if let Ok(devices) = host.output_devices() {
                    for (idx, dev) in devices.enumerate() {
                        if dev.name().ok().as_deref() == Some(name.as_str()) {
                            selected_index = Some(idx);
                            break;
                        }
                    }
                }
            }
        }
        if selected_name.is_none() {
            selected_name = default_name;
        }
        device
    };
    selected_name = device.name().ok().or(selected_name);
    settings.device_index = selected_index;
    settings.device_name = selected_name;

    let mut chosen_config: Option<cpal::SupportedStreamConfig> = None;
    if let Ok(configs) = device.supported_output_configs() {
        for cfg in configs {
            let channels_ok = settings.channels == 0 || cfg.channels() == settings.channels;
            if !channels_ok {
                continue;
            }
            if settings.sample_rate != 0 {
                let rate_ok = (cfg.min_sample_rate().0..=cfg.max_sample_rate().0)
                    .contains(&settings.sample_rate);
                if !rate_ok {
                    continue;
                }
                let candidate = cfg.with_sample_rate(cpal::SampleRate(settings.sample_rate));
                let better = match chosen_config.as_ref() {
                    None => true,
                    Some(current) => {
                        let current_rank = sample_format_rank(current.sample_format());
                        let candidate_rank = sample_format_rank(candidate.sample_format());
                        candidate_rank < current_rank
                    }
                };
                if better {
                    chosen_config = Some(candidate);
                }
            } else {
                let candidate = cfg.with_sample_rate(cfg.max_sample_rate());
                let better = match chosen_config.as_ref() {
                    None => true,
                    Some(current) => {
                        let current_rate = current.sample_rate().0;
                        let candidate_rate = candidate.sample_rate().0;
                        if candidate_rate != current_rate {
                            candidate_rate > current_rate
                        } else {
                            let current_rank = sample_format_rank(current.sample_format());
                            let candidate_rank = sample_format_rank(candidate.sample_format());
                            candidate_rank < current_rank
                        }
                    }
                };
                if better {
                    chosen_config = Some(candidate);
                }
            }
        }
    }

    let supported = if let Some(cfg) = chosen_config {
        cfg
    } else {
        device
            .default_output_config()
            .map_err(|e| set_error(e.to_string()))?
    };

    let sample_format = supported.sample_format();
    let mut config: StreamConfig = supported.into();
    if let Some(buffer_frames) = settings.buffer_frames {
        let aligned = align_output_buffer_frames(buffer_frames);
        if aligned == 0 {
            settings.buffer_frames = None;
        } else {
            settings.buffer_frames = Some(aligned);
            config.buffer_size = cpal::BufferSize::Fixed(aligned);
        }
    }

    let output_rate = config.sample_rate.0.max(1);
    // Lock mix rate to the actual output rate to avoid resampler desync/speed issues
    let mix_rate = output_rate;
    settings.sample_rate = output_rate;
    settings.channels = config.channels;

    let (tx, rx) = crossbeam_channel::bounded::<AudioCommand>(2);
    let (event_tx, event_rx) = crossbeam_channel::unbounded::<AudioEvent>();

    profiler.update_config(
        output_rate,
        config.channels,
        match config.buffer_size {
            cpal::BufferSize::Fixed(frames) => frames,
            cpal::BufferSize::Default => 0,
        },
    );

    let (
        bus_volumes,
        scene_volumes,
        head_width,
        speed_of_sound,
        streaming_buffer_frames,
        streaming_chunk_frames,
        enabled,
    ) = {
        let state = state.lock();
        (
            state.bus_volumes.clone(),
            state.scene_volumes.clone(),
            state.head_width,
            state.speed_of_sound,
            state.streaming_buffer_frames,
            state.streaming_chunk_frames,
            state.enabled,
        )
    };

    let mut engine = AudioEngine::new();
    engine.set_sample_rate(mix_rate);
    engine.set_head_width(head_width);
    engine.set_speed_of_sound(speed_of_sound);
    engine.set_streaming_config(streaming_buffer_frames, streaming_chunk_frames);
    for (bus, volume) in bus_volumes {
        engine.set_bus_volume(bus, volume);
    }
    for (scene_id, volume) in scene_volumes {
        engine.set_scene_volume(scene_id, volume);
    }
    engine.set_enabled(enabled);

    let channels = config.channels as usize;
    let use_render_thread = std::env::var("HELMER_AUDIO_RENDER_THREAD")
        .map(|value| value != "0")
        .unwrap_or(false);

    let err_fn = |err| {
        tracing::error!(target: "audio.stream", "Audio stream error: {}", err);
    };

    let (render_handle, stream_result): (AudioRenderHandle, Result<cpal::Stream, String>) =
        if use_render_thread {
            let callback_frames = match config.buffer_size {
                cpal::BufferSize::Fixed(frames) => frames as usize,
                cpal::BufferSize::Default => 1024,
            };
            let rate_ratio = mix_rate as f64 / output_rate as f64;
            let mix_callback_frames =
                ((callback_frames as f64) * rate_ratio).ceil().max(1.0) as usize;
            let block_frames = mix_callback_frames.clamp(256, 4096);
            let ring_frames = mix_callback_frames.max(block_frames).max(256) * 4;
            let ring = Arc::new(AudioRingBuffer::new(ring_frames * channels + 1));
            let render_handle = spawn_audio_renderer(
                engine,
                rx,
                event_tx.clone(),
                profiler.clone(),
                ring.clone(),
                channels,
                block_frames,
            );

            let stream_result = match sample_format {
                SampleFormat::I8 => build_stream::<i8>(
                    device,
                    &config,
                    ring.clone(),
                    profiler.clone(),
                    mix_rate,
                    output_rate,
                    err_fn,
                ),
                SampleFormat::F32 => build_stream::<f32>(
                    device,
                    &config,
                    ring.clone(),
                    profiler.clone(),
                    mix_rate,
                    output_rate,
                    err_fn,
                ),
                SampleFormat::I16 => build_stream::<i16>(
                    device,
                    &config,
                    ring.clone(),
                    profiler.clone(),
                    mix_rate,
                    output_rate,
                    err_fn,
                ),
                SampleFormat::I32 => build_stream::<i32>(
                    device,
                    &config,
                    ring.clone(),
                    profiler.clone(),
                    mix_rate,
                    output_rate,
                    err_fn,
                ),
                SampleFormat::I64 => build_stream::<i64>(
                    device,
                    &config,
                    ring.clone(),
                    profiler.clone(),
                    mix_rate,
                    output_rate,
                    err_fn,
                ),
                SampleFormat::U8 => build_stream::<u8>(
                    device,
                    &config,
                    ring.clone(),
                    profiler.clone(),
                    mix_rate,
                    output_rate,
                    err_fn,
                ),
                SampleFormat::U16 => build_stream::<u16>(
                    device,
                    &config,
                    ring.clone(),
                    profiler.clone(),
                    mix_rate,
                    output_rate,
                    err_fn,
                ),
                SampleFormat::U32 => build_stream::<u32>(
                    device,
                    &config,
                    ring.clone(),
                    profiler.clone(),
                    mix_rate,
                    output_rate,
                    err_fn,
                ),
                SampleFormat::U64 => build_stream::<u64>(
                    device,
                    &config,
                    ring.clone(),
                    profiler.clone(),
                    mix_rate,
                    output_rate,
                    err_fn,
                ),
                SampleFormat::F64 => build_stream::<f64>(
                    device,
                    &config,
                    ring.clone(),
                    profiler.clone(),
                    mix_rate,
                    output_rate,
                    err_fn,
                ),
                _ => Err(format!(
                    "Unsupported audio sample format: {:?}",
                    sample_format
                )),
            };
            (render_handle, stream_result)
        } else {
            let render_handle = AudioRenderHandle {
                stop: Arc::new(AtomicBool::new(false)),
                handle: None,
            };
            let stream_result = match sample_format {
                SampleFormat::I8 => build_stream_direct::<i8>(
                    device,
                    &config,
                    profiler.clone(),
                    engine,
                    rx,
                    event_tx.clone(),
                    err_fn,
                ),
                SampleFormat::F32 => build_stream_direct::<f32>(
                    device,
                    &config,
                    profiler.clone(),
                    engine,
                    rx,
                    event_tx.clone(),
                    err_fn,
                ),
                SampleFormat::I16 => build_stream_direct::<i16>(
                    device,
                    &config,
                    profiler.clone(),
                    engine,
                    rx,
                    event_tx.clone(),
                    err_fn,
                ),
                SampleFormat::I32 => build_stream_direct::<i32>(
                    device,
                    &config,
                    profiler.clone(),
                    engine,
                    rx,
                    event_tx.clone(),
                    err_fn,
                ),
                SampleFormat::I64 => build_stream_direct::<i64>(
                    device,
                    &config,
                    profiler.clone(),
                    engine,
                    rx,
                    event_tx.clone(),
                    err_fn,
                ),
                SampleFormat::U8 => build_stream_direct::<u8>(
                    device,
                    &config,
                    profiler.clone(),
                    engine,
                    rx,
                    event_tx.clone(),
                    err_fn,
                ),
                SampleFormat::U16 => build_stream_direct::<u16>(
                    device,
                    &config,
                    profiler.clone(),
                    engine,
                    rx,
                    event_tx.clone(),
                    err_fn,
                ),
                SampleFormat::U32 => build_stream_direct::<u32>(
                    device,
                    &config,
                    profiler.clone(),
                    engine,
                    rx,
                    event_tx.clone(),
                    err_fn,
                ),
                SampleFormat::U64 => build_stream_direct::<u64>(
                    device,
                    &config,
                    profiler.clone(),
                    engine,
                    rx,
                    event_tx.clone(),
                    err_fn,
                ),
                SampleFormat::F64 => build_stream_direct::<f64>(
                    device,
                    &config,
                    profiler.clone(),
                    engine,
                    rx,
                    event_tx.clone(),
                    err_fn,
                ),
                _ => Err(format!(
                    "Unsupported audio sample format: {:?}",
                    sample_format
                )),
            };
            (render_handle, stream_result)
        };

    let stream = match stream_result {
        Ok(stream) => stream,
        Err(err) => {
            render_handle.stop();
            return Err(set_error(err));
        }
    };

    Ok((stream, settings, tx, event_rx, enabled, render_handle))
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_audio_renderer(
    mut engine: AudioEngine,
    command_rx: Receiver<AudioCommand>,
    event_tx: Sender<AudioEvent>,
    profiler: Arc<AudioProfiler>,
    ring: Arc<AudioRingBuffer>,
    channels: usize,
    block_frames: usize,
) -> AudioRenderHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::clone(&stop);
    let handle = thread::spawn(move || {
        let mut mix_buffer: Vec<f32> = Vec::new();
        while !stop_flag.load(Ordering::Relaxed) {
            while let Ok(cmd) = command_rx.try_recv() {
                engine.apply_command(cmd);
            }

            let available_samples = ring.available_write();
            if available_samples < channels {
                thread::yield_now();
                continue;
            }

            let available_frames = available_samples / channels;
            let frames_to_mix = available_frames.min(block_frames.max(1));
            let samples_to_mix = frames_to_mix * channels;
            if mix_buffer.len() < samples_to_mix {
                mix_buffer.resize(samples_to_mix, 0.0);
            }

            let mix_start = Instant::now();
            engine.mix_into(&mut mix_buffer[..samples_to_mix], channels);
            let mix_us = mix_start.elapsed().as_micros() as u64;
            let (emitters, streaming) = engine.emitter_counts();

            profiler.mix_time_us.store(mix_us, Ordering::Relaxed);
            profiler
                .active_emitters
                .store(emitters as u32, Ordering::Relaxed);
            profiler
                .streaming_emitters
                .store(streaming as u32, Ordering::Relaxed);

            let _ = ring.push_slice(&mix_buffer[..samples_to_mix]);

            let finished = engine.take_finished_emitters();
            if !finished.is_empty() {
                let _ = event_tx.try_send(AudioEvent::EmittersFinished(finished));
            }
        }
    });

    AudioRenderHandle {
        stop,
        handle: Some(handle),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn build_stream<T>(
    device: cpal::Device,
    config: &StreamConfig,
    ring: Arc<AudioRingBuffer>,
    profiler: Arc<AudioProfiler>,
    mix_rate: u32,
    output_rate: u32,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, String>
where
    T: Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    let channels = config.channels as usize;
    let mut scratch: Vec<f32> = Vec::new();
    let mix_rate = mix_rate.max(1);
    let output_rate = output_rate.max(1);
    let resample_ratio = mix_rate as f64 / output_rate as f64;

    struct ResampleState {
        pos: f64,
        current: Vec<f32>,
        next: Vec<f32>,
        cache: Vec<f32>,
        cache_len: usize,
        cache_pos: usize,
        initialized: bool,
    }

    impl ResampleState {
        fn new(channels: usize) -> Self {
            Self {
                pos: 0.0,
                current: vec![0.0; channels.max(1)],
                next: vec![0.0; channels.max(1)],
                cache: Vec::new(),
                cache_len: 0,
                cache_pos: 0,
                initialized: false,
            }
        }

        fn fill_current(&mut self, ring: &AudioRingBuffer, channels: usize) -> bool {
            self.read_frame_into(ring, channels, true)
        }

        fn fill_next(&mut self, ring: &AudioRingBuffer, channels: usize) -> bool {
            self.read_frame_into(ring, channels, false)
        }

        fn read_frame_into(
            &mut self,
            ring: &AudioRingBuffer,
            channels: usize,
            into_current: bool,
        ) -> bool {
            let channels = channels.max(1);
            let dst = if into_current {
                &mut self.current
            } else {
                &mut self.next
            };
            if dst.len() != channels {
                dst.resize(channels, 0.0);
            }
            if self.cache_pos + channels > self.cache_len {
                const CACHE_FRAMES: usize = 512;
                let min_samples = channels * CACHE_FRAMES;
                if self.cache.len() < min_samples {
                    self.cache.resize(min_samples, 0.0);
                }
                self.cache_len = ring.pop_slice(&mut self.cache);
                self.cache_pos = 0;
            }
            if self.cache_len - self.cache_pos < channels {
                for sample in dst.iter_mut().take(channels) {
                    *sample = 0.0;
                }
                return false;
            }
            let start = self.cache_pos;
            let end = start + channels;
            dst[..channels].copy_from_slice(&self.cache[start..end]);
            self.cache_pos = end;
            true
        }
    }

    let mut resampler = ResampleState::new(channels);
    struct OutputRateTracker {
        last_instant: Instant,
        frames_accum: u64,
    }

    impl OutputRateTracker {
        fn new() -> Self {
            Self {
                last_instant: Instant::now(),
                frames_accum: 0,
            }
        }

        fn update(&mut self, now: Instant, frames: usize) -> Option<u32> {
            self.frames_accum = self.frames_accum.saturating_add(frames as u64);
            let elapsed = now.duration_since(self.last_instant);
            if elapsed < Duration::from_millis(400) {
                return None;
            }
            let secs = elapsed.as_secs_f64();
            if secs <= 0.0 {
                return None;
            }
            let rate = (self.frames_accum as f64 / secs).round() as u32;
            self.frames_accum = 0;
            self.last_instant = now;
            Some(rate)
        }
    }

    let mut rate_tracker = OutputRateTracker::new();

    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [T], _| {
                let callback_start = Instant::now();
                let frames = if channels > 0 {
                    data.len() / channels
                } else {
                    0
                };

                if resample_ratio.abs() <= f64::EPSILON || (mix_rate == output_rate) {
                    if scratch.len() < data.len() {
                        scratch.resize(data.len(), 0.0);
                    }

                    let mut filled = 0usize;
                    while filled < data.len() {
                        let written = ring.pop_slice(&mut scratch[filled..]);
                        if written == 0 {
                            break;
                        }
                        filled += written;
                    }
                    if filled < data.len() {
                        for sample in scratch[filled..].iter_mut() {
                            *sample = 0.0;
                        }
                    }

                    for (dst, src) in data.iter_mut().zip(scratch.iter()) {
                        let value = src.clamp(-1.0, 1.0);
                        *dst = T::from_sample(value);
                    }
                } else {
                    if !resampler.initialized {
                        let _ = resampler.fill_current(&ring, channels);
                        let _ = resampler.fill_next(&ring, channels);
                        resampler.pos = 0.0;
                        resampler.initialized = true;
                    }

                    let mut pos = resampler.pos;
                    for frame in 0..frames {
                        let base = frame * channels;
                        let frac = pos as f32;
                        for ch in 0..channels {
                            let a = resampler.current[ch];
                            let b = resampler.next[ch];
                            let value = (a + (b - a) * frac).clamp(-1.0, 1.0);
                            data[base + ch] = T::from_sample(value);
                        }
                        pos += resample_ratio;
                        while pos >= 1.0 {
                            pos -= 1.0;
                            resampler.current.copy_from_slice(&resampler.next);
                            let _ = resampler.fill_next(&ring, channels);
                        }
                    }
                    resampler.pos = pos;
                }

                profiler.callback_time_us.store(
                    callback_start.elapsed().as_micros() as u64,
                    Ordering::Relaxed,
                );
                profiler
                    .buffer_frames
                    .store(frames as u32, Ordering::Relaxed);
                profiler
                    .frames_mixed
                    .fetch_add(frames as u64, Ordering::Relaxed);
                if let Some(rate) = rate_tracker.update(callback_start, frames) {
                    if rate >= 8_000 && rate <= 192_000 {
                        profiler.measured_sample_rate.store(rate, Ordering::Relaxed);
                    }
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| e.to_string())?;
    Ok(stream)
}

#[cfg(not(target_arch = "wasm32"))]
fn build_stream_direct<T>(
    device: cpal::Device,
    config: &StreamConfig,
    profiler: Arc<AudioProfiler>,
    mut engine: AudioEngine,
    command_rx: Receiver<AudioCommand>,
    event_tx: Sender<AudioEvent>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, String>
where
    T: Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    let channels = config.channels as usize;
    let mut scratch: Vec<f32> = Vec::new();

    struct OutputRateTracker {
        last_instant: Instant,
        frames_accum: u64,
    }

    impl OutputRateTracker {
        fn new() -> Self {
            Self {
                last_instant: Instant::now(),
                frames_accum: 0,
            }
        }

        fn update(&mut self, now: Instant, frames: usize) -> Option<u32> {
            self.frames_accum = self.frames_accum.saturating_add(frames as u64);
            let elapsed = now.duration_since(self.last_instant);
            if elapsed < Duration::from_millis(400) {
                return None;
            }
            let secs = elapsed.as_secs_f64();
            if secs <= 0.0 {
                return None;
            }
            let rate = (self.frames_accum as f64 / secs).round() as u32;
            self.frames_accum = 0;
            self.last_instant = now;
            Some(rate)
        }
    }

    let mut rate_tracker = OutputRateTracker::new();

    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [T], _| {
                let callback_start = Instant::now();
                while let Ok(cmd) = command_rx.try_recv() {
                    engine.apply_command(cmd);
                }

                if scratch.len() < data.len() {
                    scratch.resize(data.len(), 0.0);
                }

                let mix_start = Instant::now();
                engine.mix_into(&mut scratch[..data.len()], channels);
                let mix_us = mix_start.elapsed().as_micros() as u64;
                let (emitters, streaming) = engine.emitter_counts();

                profiler.mix_time_us.store(mix_us, Ordering::Relaxed);
                profiler
                    .active_emitters
                    .store(emitters as u32, Ordering::Relaxed);
                profiler
                    .streaming_emitters
                    .store(streaming as u32, Ordering::Relaxed);

                for (dst, src) in data.iter_mut().zip(scratch.iter()) {
                    let value = src.clamp(-1.0, 1.0);
                    *dst = T::from_sample(value);
                }

                let finished = engine.take_finished_emitters();
                if !finished.is_empty() {
                    let _ = event_tx.try_send(AudioEvent::EmittersFinished(finished));
                }

                let frames = if channels > 0 {
                    data.len() / channels
                } else {
                    0
                };

                profiler.callback_time_us.store(
                    callback_start.elapsed().as_micros() as u64,
                    Ordering::Relaxed,
                );
                profiler
                    .buffer_frames
                    .store(frames as u32, Ordering::Relaxed);
                profiler
                    .frames_mixed
                    .fetch_add(frames as u64, Ordering::Relaxed);
                if let Some(rate) = rate_tracker.update(callback_start, frames) {
                    if rate >= 8_000 && rate <= 192_000 {
                        profiler.measured_sample_rate.store(rate, Ordering::Relaxed);
                    }
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| e.to_string())?;
    Ok(stream)
}

#[allow(dead_code)]
#[cfg(target_arch = "wasm32")]
pub struct AudioBackend {
    profiler: Arc<AudioProfiler>,
}

#[cfg(target_arch = "wasm32")]
impl AudioBackend {
    pub fn new() -> Self {
        Self {
            profiler: Arc::new(AudioProfiler::default()),
        }
    }

    pub fn enabled(&self) -> bool {
        false
    }

    pub fn set_enabled(&self, _enabled: bool) {}

    pub fn send_frame(
        &self,
        _listener: Option<AudioListenerSettings>,
        _emitters: Vec<AudioEmitterSnapshot>,
    ) {
    }

    pub fn set_bus_volume(&self, _bus: AudioBus, _volume: f32) {}

    pub fn bus_volume(&self, _bus: AudioBus) -> f32 {
        1.0
    }

    pub fn bus_list(&self) -> Vec<AudioBus> {
        AudioBus::DEFAULTS.to_vec()
    }

    pub fn set_scene_volume(&self, _scene_id: u64, _volume: f32) {}

    pub fn scene_volume(&self, _scene_id: u64) -> f32 {
        1.0
    }

    pub fn set_head_width(&self, _width: f32) {}

    pub fn head_width(&self) -> f32 {
        DEFAULT_HEAD_WIDTH
    }

    pub fn set_speed_of_sound(&self, _speed: f32) {}

    pub fn speed_of_sound(&self) -> f32 {
        SPEED_OF_SOUND
    }

    pub fn set_streaming_config(&self, _buffer_frames: usize, _chunk_frames: usize) {}

    pub fn streaming_config(&self) -> (usize, usize) {
        (DEFAULT_STREAM_BUFFER_FRAMES, DEFAULT_STREAM_CHUNK_FRAMES)
    }

    pub fn clear_emitters(&self) {}

    pub fn drain_finished_emitters(&self) -> Vec<u64> {
        Vec::new()
    }

    pub fn output_settings(&self) -> AudioOutputSettings {
        AudioOutputSettings::default()
    }

    pub fn last_error(&self) -> Option<String> {
        None
    }

    pub fn available_output_devices(&self) -> Vec<String> {
        Vec::new()
    }

    pub fn stats(&self) -> AudioRuntimeStats {
        AudioRuntimeStats {
            enabled: false,
            sample_rate: DEFAULT_SAMPLE_RATE,
            measured_sample_rate: DEFAULT_SAMPLE_RATE,
            channels: DEFAULT_OUTPUT_CHANNELS,
            buffer_frames: 0,
            mix_time_us: 0,
            callback_time_us: 0,
            active_emitters: 0,
            streaming_emitters: 0,
            frames_mixed: 0,
        }
    }

    pub fn reconfigure(&self, _settings: AudioOutputSettings) -> Result<(), String> {
        Ok(())
    }
}
