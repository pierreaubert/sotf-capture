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
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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
                .map(|cfg| cfg.channels() as usize)
                .unwrap_or(2)
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
#[derive(Default)]
pub struct CpalPlayback {
    stream: Option<cpal::Stream>,
    rendered_frames: Arc<AtomicUsize>,
    total_frames: usize,
}

impl std::fmt::Debug for CpalPlayback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpalPlayback")
            .field("active", &self.stream.is_some())
            .field(
                "rendered_frames",
                &self.rendered_frames.load(Ordering::Relaxed),
            )
            .field("total_frames", &self.total_frames)
            .finish()
    }
}

impl CpalPlayback {
    /// Create an idle backend; playback starts with [`SweepPlayback::start`].
    pub fn new() -> Self {
        Self::default()
    }
}

/// Read a stimulus WAV as mono f32, taking the first channel.
fn read_stimulus_mono(wav: &Path) -> Result<Vec<f32>, String> {
    let mut reader =
        hound::WavReader::open(wav).map_err(|e| format!("Failed to load file: {e}"))?;
    let spec = reader.spec();
    if spec.channels == 0 {
        return Err("Failed to load file: WAV has no channels".to_string());
    }
    let mono: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, _) => reader
            .samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .map_err(|e| format!("Failed to load file: {e}"))?
            .chunks(spec.channels as usize)
            .map(|frame| frame[0])
            .collect(),
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .collect::<Result<Vec<i16>, _>>()
            .map_err(|e| format!("Failed to load file: {e}"))?
            .chunks(spec.channels as usize)
            .map(|frame| f32::from(frame[0]) / f32::from(i16::MAX))
            .collect(),
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .collect::<Result<Vec<i32>, _>>()
            .map_err(|e| format!("Failed to load file: {e}"))?
            .chunks(spec.channels as usize)
            .map(|frame| (frame[0] as f32) / (i32::MAX as f32))
            .collect(),
        (format, bits) => {
            return Err(format!(
                "Failed to load file: unsupported WAV format {format:?} with {bits} bits"
            ));
        }
    };
    Ok(mono)
}

impl SweepPlayback for CpalPlayback {
    fn start(
        &mut self,
        wav: &Path,
        device_name: Option<&str>,
        channel: u16,
        sweep_rate: u32,
    ) -> Result<(), String> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let output_device = if let Some(name) = device_name {
            crate::devices::find_device(&host, name, false)?
        } else {
            host.default_output_device()
                .ok_or_else(|| "No default output device available".to_string())?
        };
        let hardware_channels = output_channel_count(&output_device);
        check_output_channel(channel, hardware_channels)?;

        let default_config = output_device
            .default_output_config()
            .map_err(|e| format!("Failed to query output device: {e}"))?;
        // Prefer the sweep rate when the hardware supports it at the
        // configured channel count; resampling the stimulus is exact but
        // avoiding it keeps the played signal bit-identical to the file.
        let mut device_rate = default_config.sample_rate();
        if device_rate != sweep_rate
            && let Ok(configs) = output_device.supported_output_configs()
            && configs.into_iter().any(|config| {
                config.channels() as usize == hardware_channels
                    && config.min_sample_rate() <= sweep_rate
                    && sweep_rate <= config.max_sample_rate()
            })
        {
            device_rate = sweep_rate;
        }

        let stimulus = read_stimulus_mono(wav)?;
        let stimulus = if device_rate == sweep_rate {
            stimulus
        } else {
            log::warn!(
                "[sotf-capture] output device runs at {device_rate}Hz; resampling stimulus from {sweep_rate}Hz"
            );
            super::record::resample_reference_signal(&stimulus, sweep_rate, device_rate)?
        };

        let config = cpal::StreamConfig {
            channels: hardware_channels as u16,
            sample_rate: device_rate,
            buffer_size: cpal::BufferSize::Default,
        };
        self.rendered_frames.store(0, Ordering::Relaxed);
        self.total_frames = stimulus.len();
        let rendered = Arc::clone(&self.rendered_frames);
        let channel_index = channel as usize;
        let stimulus: Arc<[f32]> = Arc::from(stimulus);
        let stream = match default_config.sample_format() {
            cpal::SampleFormat::F32 => {
                let rendered = Arc::clone(&rendered);
                let stimulus = Arc::clone(&stimulus);
                output_device.build_output_stream(
                    &config,
                    move |data: &mut [f32], _| {
                        let start =
                            rendered.fetch_add(data.len() / hardware_channels, Ordering::Relaxed);
                        for (i, frame) in data.chunks_mut(hardware_channels).enumerate() {
                            frame.fill(0.0);
                            frame[channel_index] = stimulus.get(start + i).copied().unwrap_or(0.0);
                        }
                    },
                    |e| log::error!("[sotf-capture] output stream error: {e}"),
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let rendered = Arc::clone(&rendered);
                let stimulus = Arc::clone(&stimulus);
                output_device.build_output_stream(
                    &config,
                    move |data: &mut [i16], _| {
                        let start =
                            rendered.fetch_add(data.len() / hardware_channels, Ordering::Relaxed);
                        for (i, frame) in data.chunks_mut(hardware_channels).enumerate() {
                            frame.fill(0);
                            let sample = (stimulus
                                .get(start + i)
                                .copied()
                                .unwrap_or(0.0)
                                .clamp(-1.0, 1.0)
                                * f32::from(i16::MAX))
                                as i16;
                            frame[channel_index] = sample;
                        }
                    },
                    |e| log::error!("[sotf-capture] output stream error: {e}"),
                    None,
                )
            }
            cpal::SampleFormat::U16 => {
                let rendered = Arc::clone(&rendered);
                let stimulus = Arc::clone(&stimulus);
                output_device.build_output_stream(
                    &config,
                    move |data: &mut [u16], _| {
                        let start =
                            rendered.fetch_add(data.len() / hardware_channels, Ordering::Relaxed);
                        for (i, frame) in data.chunks_mut(hardware_channels).enumerate() {
                            frame.fill(u16::MAX / 2 + 1);
                            let centered = stimulus
                                .get(start + i)
                                .copied()
                                .unwrap_or(0.0)
                                .clamp(-1.0, 1.0);
                            let sample = (centered * 32767.0 + 32768.0) as i32 as u16;
                            frame[channel_index] = sample;
                        }
                    },
                    |e| log::error!("[sotf-capture] output stream error: {e}"),
                    None,
                )
            }
            format => {
                return Err(format!("Unsupported output sample format: {format:?}"));
            }
        }
        .map_err(|e| format!("Failed to start playback: {e}"))?;
        stream
            .play()
            .map_err(|e| format!("Failed to start playback: {e}"))?;
        self.stream = Some(stream);
        Ok(())
    }

    fn is_finished(&mut self) -> bool {
        self.rendered_frames.load(Ordering::Relaxed) >= self.total_frames
    }

    fn stop(&mut self) -> Result<(), String> {
        self.stream.take();
        Ok(())
    }
}
