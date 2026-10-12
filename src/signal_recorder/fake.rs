//! Emulated audio devices for hardware-free capture tests.
//!
//! A [`VirtualWire`] connects a [`FakePlayback`] output to a
//! [`FakeRecorder`] input in memory: playback publishes the stimulus,
//! the recorder renders it through latency, clock drift, gain, and
//! seeded noise, then reports it like a cpal capture. Device lookup,
//! channel validation, and rate negotiation mirror
//! [`CpalRecorder`](super::recorder::CpalRecorder) messages so tests
//! exercise the same orchestration and error paths as production.
//!
//! Available under `cfg(test)` or the `test-util` feature (desktop
//! only); production builds never include this module.

// Rust guideline compliant 2026-02-21

use super::misc::actionable_capture_error;
use super::playback::{SweepPlayback, check_output_channel, read_stimulus_mono};
use super::recorder::{NegotiatedInput, RecordedTake, SweepRecorder, check_input_channels};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Reusable deterministic signal conditions for frontend QA fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FakeAudioProfile {
    Clean,
    HighThd,
    LowSnr,
    /// Keep audio clean but provide no noise during digital-silence windows,
    /// so the single-take SNR estimator reports unavailable evidence naturally.
    UnknownEvidence,
    Clipping,
}

/// Configuration for emulated loopback capture.
#[derive(Debug, Clone)]
pub struct FakeAudioConfig {
    /// Known input device names (exact match); empty matches nothing.
    pub input_devices: Vec<String>,
    /// Known output device names (exact match); empty matches nothing.
    pub output_devices: Vec<String>,
    /// Whether a default input device exists (`None` selector).
    pub has_default_input: bool,
    /// Whether a default output device exists (`None` selector).
    pub has_default_output: bool,
    /// Emulated hardware input channel count.
    pub input_channels: usize,
    /// Emulated hardware output channel count.
    pub output_channels: usize,
    /// Negotiated input rate; `None` accepts the requested rate.
    pub negotiated_input_rate: Option<u32>,
    /// Pure input delay prepended to the capture, in frames.
    pub latency_frames: usize,
    /// Input clock drift in ppm (positive runs fast, stretches capture).
    pub drift_ppm: f64,
    /// Linear capture gain (converter saturates at ±1.0).
    pub gain: f32,
    /// Causal room impulse response per hardware input channel (empty is identity).
    pub channel_impulse_responses: Vec<Vec<f32>>,
    /// Cubic soft-saturation coefficient applied as `tanh(x + coefficient * x^3)`.
    pub cubic_distortion: f32,
    /// Approximate RMS of the seeded room-noise floor (0 disables).
    pub noise_rms: f32,
    /// Optional noise floor applied only to digital-silence windows.
    pub quiet_window_noise_rms: f32,
    /// Noise seed; each channel adds its index (must be nonzero).
    pub seed: u64,
    /// Noise/silence tail appended after the stimulus, in seconds.
    pub tail_secs: f32,
    /// Dropped-sample count reported with the take (R6 plumbing).
    pub dropped_samples: u64,
    /// When set, every input `start` fails with this message.
    pub fail_input_start: Option<String>,
    /// When set, every input `stop` fails with this message.
    pub fail_input_stop: Option<String>,
    /// When set, every output `start` fails with this message.
    pub fail_output_start: Option<String>,
    /// First N successful input `start` calls render silence (sleeping
    /// device); later starts render the wire stimulus. Exercises the
    /// pre-capture wake-up path.
    pub dead_starts: usize,
}

impl Default for FakeAudioConfig {
    fn default() -> Self {
        Self {
            input_devices: vec!["FakeLoopback In".to_string()],
            output_devices: vec!["FakeLoopback Out".to_string()],
            has_default_input: true,
            has_default_output: true,
            input_channels: 2,
            output_channels: 2,
            negotiated_input_rate: None,
            latency_frames: 0,
            drift_ppm: 0.0,
            gain: 1.0,
            channel_impulse_responses: Vec::new(),
            cubic_distortion: 0.0,
            noise_rms: 0.0,
            quiet_window_noise_rms: 0.0,
            seed: 0x9E37_79B9_7F4A_7C15,
            tail_secs: 1.0,
            dropped_samples: 0,
            fail_input_start: None,
            fail_input_stop: None,
            fail_output_start: None,
            dead_starts: 0,
        }
    }
}

impl FakeAudioConfig {
    /// Build a seeded default configuration for a common capture QA condition.
    pub fn for_profile(profile: FakeAudioProfile) -> Self {
        let mut config = Self::default();
        match profile {
            FakeAudioProfile::Clean => {
                config.noise_rms = 0.00001;
                // A real microphone has a measurable room floor during the
                // pre-sweep silence. Keep that evidence deterministic and
                // quiet enough to pass the single-take SNR advisory.
                config.quiet_window_noise_rms = 0.00001;
            }
            FakeAudioProfile::HighThd => {
                config.noise_rms = 0.00001;
                config.cubic_distortion = 4.0;
            }
            FakeAudioProfile::LowSnr => {
                config.latency_frames = 4_800;
                config.quiet_window_noise_rms = 0.2;
            }
            FakeAudioProfile::UnknownEvidence => config.noise_rms = 0.0,
            FakeAudioProfile::Clipping => {
                config.noise_rms = 0.00001;
                config.gain = 2.0;
            }
        }
        config
    }
}

/// Stimulus published by [`FakePlayback`] for [`FakeRecorder`].
#[derive(Debug, Default)]
struct WireState {
    stimulus: Vec<f32>,
    sweep_rate: u32,
    play_count: u64,
    output_channel: u16,
    output_device: Option<String>,
    played: bool,
}

/// Shared loopback wire between fake playback and fake capture.
///
/// Cloning shares the same underlying stimulus; hand one clone to
/// each backend. Inspectors (`played_stimulus`) support assertions.
#[derive(Debug, Clone)]
pub struct VirtualWire {
    state: Arc<Mutex<WireState>>,
    config: FakeAudioConfig,
}

impl Default for VirtualWire {
    fn default() -> Self {
        Self::new(FakeAudioConfig::default())
    }
}

impl VirtualWire {
    /// Create a wire with the given device/signal configuration.
    pub fn new(config: FakeAudioConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(WireState::default())),
            config,
        }
    }

    /// Create a wire with one of the standard deterministic QA conditions.
    pub fn for_profile(profile: FakeAudioProfile) -> Self {
        Self::new(FakeAudioConfig::for_profile(profile))
    }

    /// Set a room impulse response for one hardware input channel.
    pub fn with_channel_impulse_response(
        mut self,
        channel: usize,
        impulse_response: Vec<f32>,
    ) -> Self {
        if self.config.channel_impulse_responses.len() <= channel {
            self.config
                .channel_impulse_responses
                .resize_with(channel + 1, Vec::new);
        }
        self.config.channel_impulse_responses[channel] = impulse_response;
        self
    }

    /// Add controlled cubic distortion before converter saturation.
    pub fn with_cubic_distortion(mut self, coefficient: f32) -> Self {
        self.config.cubic_distortion = coefficient;
        self
    }

    /// Build the output side of this wire.
    pub fn playback(&self, log_tag: impl Into<String>) -> FakePlayback {
        FakePlayback {
            log_tag: log_tag.into(),
            wire: self.clone(),
            rendered_frames: 0,
            total_frames: 0,
        }
    }

    /// Build the input side of this wire.
    pub fn recorder(&self, log_tag: impl Into<String>) -> FakeRecorder {
        FakeRecorder {
            log_tag: log_tag.into(),
            wire: self.clone(),
            capture: None,
            starts: 0,
        }
    }

    /// Stimulus published so far, if playback started.
    pub fn played_stimulus(&self) -> Option<Vec<f32>> {
        let state = self.state.lock().ok()?;
        state.played.then(|| state.stimulus.clone())
    }
}

/// In-memory [`SweepPlayback`] publishing to a [`VirtualWire`].
///
/// Device lookup is exact-name-only (unlike cpal priority matching);
/// playback itself completes instantly.
#[derive(Debug)]
pub struct FakePlayback {
    log_tag: String,
    wire: VirtualWire,
    rendered_frames: usize,
    total_frames: usize,
}

impl FakePlayback {
    /// Resolve the output selector against the emulated registry.
    fn resolve_device(&self, device: Option<&str>) -> Result<Option<String>, String> {
        let config = &self.wire.config;
        match device {
            None if config.has_default_output => Ok(None),
            None => Err("No default output device available".to_string()),
            Some(name) if config.output_devices.iter().any(|d| d == name) => {
                Ok(Some(name.to_string()))
            }
            Some(name) => {
                let info: Vec<(String, String)> = config
                    .output_devices
                    .iter()
                    .map(|d| (d.clone(), d.clone()))
                    .collect();
                Err(crate::devices::device_not_found_message(name, false, &info))
            }
        }
    }
}

impl SweepPlayback for FakePlayback {
    fn start(
        &mut self,
        wav: &Path,
        device: Option<&str>,
        channel: u16,
        sweep_rate: u32,
    ) -> Result<(), String> {
        let resolved = self.resolve_device(device)?;
        check_output_channel(channel, self.wire.config.output_channels)?;
        if let Some(failure) = &self.wire.config.fail_output_start {
            return Err(failure.clone());
        }
        let stimulus = read_stimulus_mono(wav)?;
        let mut state = self
            .wire
            .state
            .lock()
            .map_err(|_| "Fake playback wire is poisoned".to_string())?;
        self.total_frames = stimulus.len();
        // Instant playback: the stimulus is fully rendered at start, so
        // the orchestration poll loop exits on its first tick.
        self.rendered_frames = self.total_frames;
        state.stimulus = stimulus;
        state.sweep_rate = sweep_rate;
        state.play_count = state.play_count.wrapping_add(1);
        state.output_channel = channel;
        state.output_device = resolved;
        state.played = true;
        log::info!(
            "[{}] Fake playback: {} frames to output channel {channel}",
            self.log_tag,
            self.total_frames,
        );
        Ok(())
    }

    fn is_finished(&mut self) -> bool {
        self.rendered_frames >= self.total_frames
    }

    fn stop(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// Active fake capture between `start` and `stop`.
#[derive(Debug)]
struct FakeCapture {
    channels: Vec<u16>,
    sample_rate: u32,
    expected_duration_secs: f64,
}

/// In-memory [`SweepRecorder`] rendering from a [`VirtualWire`].
///
/// Renders the published stimulus through rate conversion, clock
/// drift, latency, gain, converter saturation, and seeded noise.
/// Device/channel/rate errors mirror [`CpalRecorder`](super::recorder::CpalRecorder).
#[derive(Debug)]
pub struct FakeRecorder {
    log_tag: String,
    wire: VirtualWire,
    capture: Option<FakeCapture>,
    starts: usize,
}

impl FakeRecorder {
    /// Tail length in frames at the capture rate.
    fn tail_frames(&self, sample_rate: u32) -> usize {
        (self.wire.config.tail_secs.max(0.0) * sample_rate as f32).round() as usize
    }

    /// Render one channel through rate conversion, room response, gain, distortion, and noise.
    fn render_channel(
        &self,
        capture: &FakeCapture,
        channel_index: usize,
    ) -> Result<Vec<f32>, String> {
        let config = &self.wire.config;
        let (stimulus, sweep_rate, play_count) = match self.wire.state.lock() {
            Ok(state) if state.played && self.starts > config.dead_starts => {
                (state.stimulus.clone(), state.sweep_rate, state.play_count)
            }
            _ => (Vec::new(), capture.sample_rate, 0),
        };
        let mut signal = if stimulus.is_empty() || sweep_rate == capture.sample_rate {
            stimulus
        } else {
            super::record::resample_reference_signal(&stimulus, sweep_rate, capture.sample_rate)?
        };
        if signal.is_empty() {
            // No playback (or an empty stimulus) captures room tone only,
            // so the silence gate downstream behaves as on hardware.
            let frames = (capture.expected_duration_secs.max(0.0) * f64::from(capture.sample_rate))
                .ceil() as usize
                + self.tail_frames(capture.sample_rate);
            signal = vec![0.0; frames];
        } else {
            signal = warp_drift(&signal, config.drift_ppm);
            signal.extend(std::iter::repeat_n(
                0.0,
                self.tail_frames(capture.sample_rate),
            ));
        }
        if let Some(impulse_response) = config
            .channel_impulse_responses
            .get(channel_index)
            .filter(|response| !response.is_empty())
        {
            if impulse_response.iter().any(|tap| !tap.is_finite()) {
                return Err(format!(
                    "Fake room impulse response for input channel {channel_index} contains non-finite samples"
                ));
            }
            signal = convolve_causal(&signal, impulse_response);
        }
        if config.latency_frames > 0 {
            let mut delayed = vec![0.0; config.latency_frames];
            delayed.append(&mut signal);
            signal = delayed;
        }
        if config.gain != 1.0 {
            for sample in &mut signal {
                *sample *= config.gain;
            }
        }
        if config.cubic_distortion != 0.0 {
            if !config.cubic_distortion.is_finite() {
                return Err("Fake cubic distortion coefficient must be finite".to_string());
            }
            for sample in &mut signal {
                let input = *sample;
                *sample = (input + config.cubic_distortion * input.powi(3)).tanh();
            }
        }
        // Converter saturation ahead of the noise floor, as in hardware.
        for sample in &mut signal {
            *sample = sample.clamp(-1.0, 1.0);
        }
        if config.noise_rms > 0.0 || config.quiet_window_noise_rms > 0.0 {
            // Keep every run reproducible while giving consecutive repeated
            // sweeps independent room-noise realizations for coherence/SNR.
            let seed = config
                .seed
                .wrapping_add(channel_index as u64)
                .wrapping_add(play_count.wrapping_mul(0xD1B5_4A32_D192_ED03));
            let mut rng = XorShift64(seed.max(1));
            // Uniform(-1, 1) has RMS 1/sqrt(3); scale to the target RMS.
            let amplitude = config.noise_rms * 3.0_f32.sqrt();
            let quiet_amplitude = config.quiet_window_noise_rms * 3.0_f32.sqrt();
            for sample in &mut signal {
                let noise = if sample.abs() <= f32::EPSILON {
                    quiet_amplitude
                } else {
                    amplitude
                };
                *sample = (*sample + rng.next_unit() * noise).clamp(-1.0, 1.0);
            }
        }
        Ok(signal)
    }
}

impl SweepRecorder for FakeRecorder {
    fn start(
        &mut self,
        device: Option<&str>,
        channels: &[u16],
        sample_rate: u32,
        expected_duration_secs: f64,
    ) -> Result<NegotiatedInput, String> {
        let tag = self.log_tag.as_str();
        if channels.is_empty() {
            return Err(format!("[{tag}] No input channels requested"));
        }
        let config = &self.wire.config;
        match device {
            None if config.has_default_input => {}
            None => {
                return Err(actionable_capture_error(
                    &format!("[{tag}]"),
                    &"No default input device available",
                ));
            }
            Some(name) if config.input_devices.iter().any(|d| d == name) => {}
            Some(name) => {
                let info: Vec<(String, String)> = config
                    .input_devices
                    .iter()
                    .map(|d| (d.clone(), d.clone()))
                    .collect();
                return Err(actionable_capture_error(
                    &format!("[{tag}] Input device not usable"),
                    &crate::devices::device_not_found_message(name, true, &info),
                ));
            }
        }
        check_input_channels(channels, config.input_channels)?;
        let negotiated = config.negotiated_input_rate.unwrap_or(sample_rate);
        if negotiated != sample_rate {
            log::warn!(
                "[{tag}] INPUT SAMPLE RATE MISMATCH: recording at {negotiated}Hz but sweep/reference at {sample_rate}Hz",
            );
        }
        if let Some(failure) = &config.fail_input_start {
            return Err(failure.clone());
        }
        log::info!(
            "[{tag}] Recording from input channels {channels:?} (0-indexed) out of {} total channels at {negotiated}Hz",
            config.input_channels,
        );
        self.capture = Some(FakeCapture {
            channels: channels.to_vec(),
            sample_rate: negotiated,
            expected_duration_secs,
        });
        self.starts += 1;
        Ok(NegotiatedInput {
            hardware_channels: config.input_channels,
            sample_rate: negotiated,
        })
    }

    fn captured_samples(&self) -> usize {
        let Some(capture) = &self.capture else {
            return 0;
        };
        let Ok(state) = self.wire.state.lock() else {
            return 0;
        };
        if !state.played || state.sweep_rate == 0 {
            return 0;
        }
        // Progress-only estimate; drift warping (±0.1%) is ignored here.
        (state.stimulus.len() as f64 * f64::from(capture.sample_rate) / f64::from(state.sweep_rate))
            .ceil() as usize
            + self.wire.config.latency_frames
            + capture
                .channels
                .iter()
                .filter_map(|channel| {
                    self.wire
                        .config
                        .channel_impulse_responses
                        .get(usize::from(*channel))
                })
                .map(|response| response.len().saturating_sub(1))
                .max()
                .unwrap_or(0)
            + self.tail_frames(capture.sample_rate)
    }

    fn stop(&mut self) -> Result<RecordedTake, String> {
        let Some(capture) = self.capture.take() else {
            return Err(format!(
                "[{}] Recorder stopped without a matching start",
                self.log_tag
            ));
        };
        if let Some(failure) = &self.wire.config.fail_input_stop {
            return Err(failure.clone());
        }
        let mut recorded_per_channel = Vec::with_capacity(capture.channels.len());
        for &input_channel in &capture.channels {
            recorded_per_channel.push(self.render_channel(&capture, usize::from(input_channel))?);
        }
        let dropped_samples = self.wire.config.dropped_samples;
        if dropped_samples > 0 {
            log::warn!(
                "[{}] Dropped {dropped_samples} input samples because the capture ring buffer filled",
                self.log_tag
            );
        }
        Ok(RecordedTake {
            recorded_per_channel,
            analysis_sample_rate: capture.sample_rate,
            dropped_samples,
        })
    }
}

/// Time-warp samples by input clock drift (linear interpolation).
///
/// Positive `drift_ppm` stretches the capture (fast input clock);
/// zero returns the input unchanged. Absurd factors fall back to
/// unwarped rather than fabricating garbage lengths.
fn warp_drift(samples: &[f32], drift_ppm: f64) -> Vec<f32> {
    if drift_ppm == 0.0 || samples.is_empty() {
        return samples.to_vec();
    }
    let factor = 1.0 + drift_ppm / 1_000_000.0;
    if !factor.is_finite() || !(0.0 < factor && factor < 10.0) {
        return samples.to_vec();
    }
    let out_len = ((samples.len() as f64 * factor).round() as usize).max(1);
    (0..out_len)
        .map(|i| {
            let position = i as f64 / factor;
            let low = position.floor() as usize;
            let frac = (position - low as f64) as f32;
            let a = samples.get(low).copied().unwrap_or(0.0);
            let b = samples.get(low + 1).copied().unwrap_or(0.0);
            a + (b - a) * frac
        })
        .collect()
}

/// Apply a causal room response while retaining the response's full decay.
fn convolve_causal(signal: &[f32], impulse_response: &[f32]) -> Vec<f32> {
    if signal.is_empty() || impulse_response.is_empty() {
        return signal.to_vec();
    }
    let output_len = signal.len() + impulse_response.len() - 1;
    let mut output = vec![0.0; output_len];
    for (sample_index, &sample) in signal.iter().enumerate() {
        for (tap_index, &tap) in impulse_response.iter().enumerate() {
            output[sample_index + tap_index] += sample * tap;
        }
    }
    output
}

/// Deterministic 64-bit xorshift (test noise needs no RNG dependency).
struct XorShift64(u64);

impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform sample in [-1, 1).
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1_u64 << 31) as f32 - 1.0
    }
}
