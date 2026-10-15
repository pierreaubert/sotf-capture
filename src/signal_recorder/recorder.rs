//! Mono capture backends for measurement takes.
//!
//! Capture records one hardware input channel per microphone while a
//! separate [`SweepPlayback`](super::playback::SweepPlayback) backend plays
//! the stimulus. Backends own only the input side: device lookup,
//! channel and rate negotiation, stream lifetime, and ring draining.
//!
//! [`CpalRecorder`] is the production backend used by the sweep capture
//! entry points. Emulated backends live behind the `test-util` feature
//! (see `super::fake`) so hardware-free tests run the same orchestration
//! (pre-roll, polling, tail capture, gating) as production.

// Rust guideline compliant 2026-02-21

#[cfg(feature = "capture-io")]
use std::sync::Arc;
#[cfg(feature = "capture-io")]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Input configuration negotiated with the device at capture start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegotiatedInput {
    /// Hardware input channel count of the opened stream.
    pub hardware_channels: usize,
    /// Input rate actually negotiated; may differ from requested.
    pub sample_rate: u32,
}

/// One drained capture: mono samples per requested channel.
#[derive(Debug)]
pub struct RecordedTake {
    /// Raw mono captures in requested-channel order (not drift-corrected).
    pub recorded_per_channel: Vec<Vec<f32>>,
    /// Input sample rate the samples were captured at.
    pub analysis_sample_rate: u32,
    /// Input samples dropped because capture rings filled (R6).
    pub dropped_samples: u64,
}

/// Mono capture on selected hardware input channels.
///
/// Implementations open the named input device (or the default one),
/// negotiate a stream covering `channels` at `sample_rate` when the
/// hardware supports it, and capture until [`SweepRecorder::stop`].
/// Each successful `start` must be followed by exactly one `stop`; a
/// failed `start` needs no matching `stop`.
pub trait SweepRecorder {
    /// Open the device and start capturing `channels`.
    ///
    /// # Errors
    ///
    /// Returns a message when no channels are requested, the device is
    /// unavailable, a channel is out of range, or the stream fails.
    fn start(
        &mut self,
        device: Option<&str>,
        channels: &[u16],
        sample_rate: u32,
        expected_duration_secs: f64,
    ) -> Result<NegotiatedInput, String>;

    /// Samples captured on the first requested channel (progress only).
    fn captured_samples(&self) -> usize;

    /// Halt capture and return the drained take.
    ///
    /// # Errors
    ///
    /// Returns a message when capture never started or cannot stop cleanly.
    fn stop(&mut self) -> Result<RecordedTake, String>;
}

/// Validate requested input channels against hardware capacity.
///
/// # Errors
///
/// Returns a message naming the first offending channel and capacity.
pub fn check_input_channels(channels: &[u16], hardware_channels: usize) -> Result<(), String> {
    if let Some(&channel) = channels
        .iter()
        .find(|channel| (**channel as usize) >= hardware_channels)
    {
        return Err(format!(
            "Input channel {channel} exceeds hardware channel count {hardware_channels} (channels are 0-indexed)"
        ));
    }
    Ok(())
}

/// cpal-native [`SweepRecorder`] without emulation.
///
/// Opens a cpal input stream at the requested rate when the device
/// supports it; otherwise falls back to the device default rate and
/// reports it so the caller resamples the reference.
#[cfg(feature = "capture-io")]
pub struct CpalRecorder {
    log_tag: String,
    stream: Option<cpal::Stream>,
    consumers: Vec<rtrb::Consumer<f32>>,
    counts: Vec<Arc<AtomicUsize>>,
    overruns: Arc<AtomicUsize>,
    sample_rate: u32,
}

#[cfg(feature = "capture-io")]
impl std::fmt::Debug for CpalRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpalRecorder")
            .field("active", &self.stream.is_some())
            .field("channels", &self.consumers.len())
            .field("sample_rate", &self.sample_rate)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "capture-io")]
impl CpalRecorder {
    /// Create an idle recorder; capture starts with [`SweepRecorder::start`].
    pub fn new(log_tag: impl Into<String>) -> Self {
        Self {
            log_tag: log_tag.into(),
            stream: None,
            consumers: Vec::new(),
            counts: Vec::new(),
            overruns: Arc::new(AtomicUsize::new(0)),
            sample_rate: 0,
        }
    }
}

#[cfg(feature = "capture-io")]
impl SweepRecorder for CpalRecorder {
    #[expect(
        clippy::too_many_lines,
        reason = "Keep the ordered acquisition and artifact-processing phases together so cleanup and provenance remain explicit."
    )]
    fn start(
        &mut self,
        device: Option<&str>,
        channels: &[u16],
        sample_rate: u32,
        expected_duration_secs: f64,
    ) -> Result<NegotiatedInput, String> {
        use super::misc::{actionable_capture_error, capture_capacity};
        use super::write::write_selected_channel_to_ring;
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let tag = self.log_tag.as_str();
        if channels.is_empty() {
            return Err(format!("[{tag}] No input channels requested"));
        }

        let host = cpal::default_host();
        let input_device = if let Some(dev_name) = device {
            log::info!("[{tag}] Looking for input device: {dev_name}");
            crate::devices::find_device(&host, dev_name, true).map_err(|e| {
                actionable_capture_error(&format!("[{tag}] Input device not usable"), &e)
            })?
        } else {
            log::debug!("[{tag}] Using default input device");
            host.default_input_device().ok_or_else(|| {
                actionable_capture_error(&format!("[{tag}]"), &"No default input device available")
            })?
        };

        log::info!(
            "[{tag}] Input device: {}",
            input_device
                .description()
                .map_or_else(|_| "Unknown Device".to_string(), |d| d.name().to_string())
        );

        let default_input_config = input_device.default_input_config().map_err(|e| {
            actionable_capture_error(&format!("[{tag}] Failed to get default input config"), &e)
        })?;

        let default_input_channels = default_input_config.channels() as usize;
        let default_input_sample_rate = default_input_config.sample_rate();

        log::info!(
            "[{tag}] Input device default config: {default_input_channels}ch, {default_input_sample_rate}Hz",
        );

        // Prefer a supported config matching the requested rate with
        // enough channels (fewest channels first); fall back to default.
        let min_channels_needed = channels.iter().copied().max().unwrap_or(0) as usize + 1;
        let best_config = input_device
            .supported_input_configs()
            .ok()
            .and_then(|configs| {
                configs
                    .filter(|c| {
                        let ch = c.channels() as usize;
                        ch >= min_channels_needed
                            && c.min_sample_rate() <= sample_rate
                            && c.max_sample_rate() >= sample_rate
                    })
                    .min_by_key(cpal::SupportedStreamConfigRange::channels)
            });

        let (hardware_input_channels, input_sample_rate) = if let Some(config) = best_config {
            let ch = config.channels() as usize;
            log::info!(
                "[{tag}] Using supported config: {ch}ch, {sample_rate}Hz (requested {sample_rate}Hz)",
            );
            (ch, sample_rate)
        } else {
            log::warn!(
                "[{tag}] No supported config for {min_channels_needed}ch at {sample_rate}Hz, using default ({default_input_channels}ch, {default_input_sample_rate}Hz)",
            );
            (default_input_channels, default_input_sample_rate)
        };

        if input_sample_rate != sample_rate {
            log::warn!(
                "[{tag}] INPUT SAMPLE RATE MISMATCH: recording at {input_sample_rate}Hz but sweep/reference at {sample_rate}Hz",
            );
        }

        check_input_channels(channels, hardware_input_channels)?;

        let input_config = cpal::StreamConfig {
            channels: u16::try_from(hardware_input_channels)
                .map_err(|_| "hardware input channel count exceeds u16".to_string())?,
            sample_rate: input_sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        log::info!(
            "[{tag}] Recording from input channels {channels:?} (0-indexed) out of {hardware_input_channels} total channels at {input_sample_rate}Hz",
        );

        // One lock-free ring per channel plus a shared overrun counter.
        let ring_capacity = capture_capacity(input_sample_rate, expected_duration_secs, 4.5);
        let mut producers = Vec::with_capacity(channels.len());
        let mut consumers = Vec::with_capacity(channels.len());
        let mut counts = Vec::with_capacity(channels.len());
        for _ in channels {
            let (producer, consumer) = rtrb::RingBuffer::<f32>::new(ring_capacity);
            producers.push(producer);
            consumers.push(consumer);
            counts.push(Arc::new(AtomicUsize::new(0)));
        }
        let counts_callback: Vec<Arc<AtomicUsize>> = counts.iter().map(Arc::clone).collect();
        let overruns = Arc::new(AtomicUsize::new(0));
        let overruns_callback = Arc::clone(&overruns);
        let channels_vec: Vec<usize> = channels.iter().map(|&c| c as usize).collect();
        let overrun_tag = self.log_tag.clone();
        let error_tag = self.log_tag.clone();

        let stream = input_device
            .build_input_stream(
                &input_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let frames = data.len() / hardware_input_channels;
                    for (mic_i, &ch_idx) in channels_vec.iter().enumerate() {
                        let written = write_selected_channel_to_ring(
                            &mut producers[mic_i],
                            data,
                            hardware_input_channels,
                            ch_idx,
                        );
                        counts_callback[mic_i].fetch_add(written, Ordering::Relaxed);
                        if written < frames {
                            // Count the actual dropped samples, not just the
                            // number of callbacks that saw a shortfall.
                            overruns_callback.fetch_add(frames - written, Ordering::Relaxed);
                            crate::rate_limited_log!(
                                warn,
                                5,
                                "[{overrun_tag}] Input capture ring buffer overrun"
                            );
                        }
                    }
                },
                move |err| {
                    crate::rate_limited_log!(warn, 5, "[{error_tag}] Input stream error: {err}");
                },
                None,
            )
            .map_err(|e| {
                actionable_capture_error(&format!("[{tag}] Failed to build input stream"), &e)
            })?;

        stream.play().map_err(|e| {
            actionable_capture_error(&format!("[{tag}] Failed to start input stream"), &e)
        })?;
        log::debug!("[{tag}] Recording started");

        self.stream = Some(stream);
        self.consumers = consumers;
        self.counts = counts;
        self.overruns = overruns;
        self.sample_rate = input_sample_rate;

        Ok(NegotiatedInput {
            hardware_channels: hardware_input_channels,
            sample_rate: input_sample_rate,
        })
    }

    fn captured_samples(&self) -> usize {
        self.counts
            .first()
            .map_or(0, |count| count.load(Ordering::Relaxed))
    }

    fn stop(&mut self) -> Result<RecordedTake, String> {
        use super::misc::drain_capture;

        if self.stream.take().is_none() {
            return Err(format!(
                "[{}] Recorder stopped without a matching start",
                self.log_tag
            ));
        }
        let consumers = std::mem::take(&mut self.consumers);
        let counts = std::mem::take(&mut self.counts);
        let mut recorded_per_channel = Vec::with_capacity(consumers.len());
        for (mut consumer, count) in consumers.into_iter().zip(counts.iter()) {
            recorded_per_channel.push(drain_capture(&mut consumer, count.load(Ordering::Relaxed)));
        }
        let dropped_samples = self.overruns.load(Ordering::Relaxed);
        if dropped_samples > 0 {
            log::warn!(
                "[{}] Dropped {dropped_samples} input samples because the capture ring buffer filled",
                self.log_tag
            );
        }
        Ok(RecordedTake {
            recorded_per_channel,
            analysis_sample_rate: self.sample_rate,
            dropped_samples: dropped_samples as u64,
        })
    }
}
