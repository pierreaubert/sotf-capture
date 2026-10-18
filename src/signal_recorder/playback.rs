//! Sweep playback backends for capture takes.
//!
//! Capture plays a mono stimulus on one hardware output channel while a
//! separate cpal input stream records. Input capture stays in the caller;
//! backends own only the output side: device lookup, channel routing,
//! stimulus resampling, and completion reporting.
//!
//! [`CpalPlayback`] is the engine-free backend used by the `sotf-capture`
//! CLI and new callers. The DAW-engine backend lives in `sotf-engine`
//! (which implements [`SweepPlayback`] over `AudioEngineManager`) so
//! existing `sotf` frontends keep byte-identical playback behavior.

// Rust guideline compliant 2026-02-21

use std::path::Path;
#[cfg(feature = "capture-io")]
use std::sync::Arc;
#[cfg(feature = "capture-io")]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Mono stimulus playback on one hardware output channel.
///
/// Implementations open the named output device (or the default one),
/// route a mono stimulus to `channel`, resampling from `sweep_rate` when
/// the device runs at another rate, and report completion. All methods
/// are synchronous; `start` blocks only for setup, never for playback.
pub trait SweepPlayback {
    /// Start routed playback of the stimulus WAV.
    ///
    /// # Errors
    ///
    /// Returns a message when the device is unavailable, the channel is
    /// out of range, the WAV cannot be read, or the output stream fails.
    fn start(
        &mut self,
        wav: &Path,
        device: Option<&str>,
        channel: u16,
        sweep_rate: u32,
    ) -> Result<(), String>;

    /// Pump backend events; true once the stimulus fully played.
    fn is_finished(&mut self) -> bool;

    /// Report latched asynchronous stream errors before accepting completion.
    ///
    /// Backends with asynchronous errors must override this method.
    ///
    /// # Errors
    /// Returns a message when playback failed even if all samples were submitted.
    fn check_health(&self) -> Result<(), String> {
        Ok(())
    }

    /// Stop playback; succeeds when silent afterwards (idempotent).
    ///
    /// # Errors
    ///
    /// Returns a message when the backend cannot stop cleanly.
    fn stop(&mut self) -> Result<(), String>;
}

/// Hardware output channel count for stream configuration.
///
/// Uses the maximum across supported configs (not just the default,
/// which can under-report hardware capability), falling back to the
/// default config and finally to stereo. Shared by every backend so
/// device-mode selection cannot drift between implementations.
#[cfg(feature = "capture-io")]
#[must_use]
pub fn output_channel_count(device: &cpal::Device) -> usize {
    use cpal::traits::DeviceTrait;

    device
        .supported_output_configs()
        .map(|configs| configs.map(|config| config.channels() as usize).max())
        .ok()
        .flatten()
        .unwrap_or_else(|| {
            device
                .default_output_config()
                .map_or(2, |cfg| cfg.channels() as usize)
        })
}

/// Validate a 0-indexed output channel against hardware capacity.
///
/// # Errors
///
/// Returns a message naming the offending channel and capacity.
pub fn check_output_channel(channel: u16, hardware_channels: usize) -> Result<(), String> {
    if (channel as usize) >= hardware_channels {
        return Err(format!(
            "Output channel {channel} exceeds hardware channel count {hardware_channels} (channels are 0-indexed)"
        ));
    }
    Ok(())
}

/// cpal-native [`SweepPlayback`] without the DAW engine.
///
/// Opens a cpal output stream at the sweep rate when the device
/// supports it (avoiding stimulus resampling); otherwise resamples the
/// stimulus offline to the device default rate with the same sinc
/// parameters as the engine resampler preset used for references.
#[cfg(feature = "capture-io")]
#[derive(Default)]
pub struct CpalPlayback {
    stream: Option<cpal::Stream>,
    rendered_frames: Arc<AtomicUsize>,
    total_frames: usize,
    failed: Arc<AtomicBool>,
    presentation: Option<Arc<super::presentation::PresentationClock>>,
}

#[cfg(feature = "capture-io")]
impl std::fmt::Debug for CpalPlayback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpalPlayback")
            .field("active", &self.stream.is_some())
            .field(
                "rendered_frames",
                &self.rendered_frames.load(Ordering::Relaxed),
            )
            .field("total_frames", &self.total_frames)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "capture-io")]
impl CpalPlayback {
    /// Create an idle backend; playback starts with [`SweepPlayback::start`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Read a stimulus WAV as mono f32, taking the first channel.
#[expect(
    clippy::cast_precision_loss,
    reason = "Integer PCM is intentionally normalized to f32 audio samples."
)]
#[cfg(any(test, feature = "test-util", feature = "capture-io"))]
pub(crate) fn read_stimulus_mono(wav: &Path) -> Result<Vec<f32>, String> {
    let file = crate::artifacts::open_read(wav)?;
    if file.metadata().map_err(|e| e.to_string())?.len() > crate::artifacts::MAX_AUDIO_BYTES as u64
    {
        return Err("stimulus WAV exceeds its size limit".into());
    }
    let mut reader = hound::WavReader::new(std::io::BufReader::new(file))
        .map_err(|e| format!("Failed to load file: {e}"))?;
    let spec = reader.spec();
    if spec.channels == 0
        || spec.channels > 64
        || reader.duration() == 0
        || reader.duration() as usize > crate::limits::MAX_CAPTURE_FRAMES
    {
        return Err("invalid stimulus WAV channel count or duration".into());
    }
    let mut mono = Vec::new();
    mono.try_reserve_exact(reader.duration() as usize)
        .map_err(|e| e.to_string())?;
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => {
            for (index, sample) in reader.samples::<f32>().enumerate() {
                let sample = sample.map_err(|e| e.to_string())?;
                if !sample.is_finite() {
                    return Err("stimulus WAV contains non-finite audio".into());
                }
                if index % spec.channels as usize == 0 {
                    mono.push(sample);
                }
            }
        }
        (hound::SampleFormat::Int, bits @ 1..=32) => {
            let scale = 2.0_f32.powi(i32::from(bits) - 1);
            for (index, sample) in reader.samples::<i32>().enumerate() {
                let sample = sample.map_err(|e| e.to_string())? as f32 / scale;
                if index % spec.channels as usize == 0 {
                    mono.push(sample);
                }
            }
        }
        (format, bits) => return Err(format!("unsupported stimulus WAV format {format:?}/{bits}")),
    }

    Ok(mono)
}

#[cfg(feature = "capture-io")]
impl SweepPlayback for CpalPlayback {
    fn start(
        &mut self,
        wav: &Path,
        device_name: Option<&str>,
        channel: u16,
        sweep_rate: u32,
    ) -> Result<(), String> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        if self.stream.is_some() {
            return Err("playback is already active".into());
        }
        let host = crate::devices::get_host_for_device(device_name);
        let output_device = if let Some(name) = device_name {
            crate::devices::find_device(&host, name, false)?
        } else {
            host.default_output_device()
                .ok_or_else(|| "No default output device available".to_string())?
        };
        let default_config = output_device
            .default_output_config()
            .map_err(|e| e.to_string())?;
        let supported: Vec<_> = output_device
            .supported_output_configs()
            .map_err(|e| e.to_string())?
            .collect();
        let min_channels = channel.checked_add(1).ok_or("output channel overflow")?;
        let selected =
            super::choose::choose_measurement_output_config(&supported, min_channels, sweep_rate)
                .or_else(|_| {
                super::choose::choose_measurement_output_config(
                    &supported,
                    min_channels,
                    default_config.sample_rate(),
                )
            })?;
        let hardware_channels = selected.channels as usize;
        let device_rate = selected.sample_rate;
        check_output_channel(channel, hardware_channels)?;

        let stimulus = read_stimulus_mono(wav)?;
        let stimulus = if device_rate == sweep_rate {
            stimulus
        } else {
            log::warn!(
                "[sotf-capture] output device runs at {device_rate}Hz; resampling stimulus from {sweep_rate}Hz"
            );
            super::record::resample_reference_signal(&stimulus, sweep_rate, device_rate)?
        };

        self.rendered_frames.store(0, Ordering::Relaxed);
        self.failed.store(false, Ordering::Release);
        self.total_frames = stimulus.len();
        let playback = Arc::new(super::build::SequentialPlayback {
            signal: stimulus,
            routes: vec![channel],
            silence_frames: 0,
            channels: hardware_channels,
            total_frames: self.total_frames,
        });
        let presentation = Arc::new(super::presentation::PresentationClock::new());
        let stream = super::build::build_measurement_output_stream(
            &output_device,
            selected,
            playback,
            Arc::clone(&self.rendered_frames),
            Arc::clone(&self.failed),
            Arc::clone(&presentation),
            "sotf-capture",
        )?;
        stream
            .play()
            .map_err(|e| format!("Failed to start playback: {e}"))?;
        self.stream = Some(stream);
        self.presentation = Some(presentation);
        Ok(())
    }

    fn is_finished(&mut self) -> bool {
        self.presentation
            .as_ref()
            .is_some_and(|clock| clock.is_finished())
    }

    fn check_health(&self) -> Result<(), String> {
        if self.failed.load(Ordering::Acquire) {
            Err("output audio stream failed".into())
        } else {
            Ok(())
        }
    }

    fn stop(&mut self) -> Result<(), String> {
        self.stream.take();
        self.presentation = None;
        self.check_health()
    }
}
