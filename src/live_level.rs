//! Bounded raw level and RTA monitoring outside the audio callback.
//!
//! Levels use unweighted RMS dBFS. The spectrum uses the existing math-dsp
//! symmetric-Hann, one-sided peak-amplitude estimator, with no overlap or padding.
//! Microphone response and absolute SPL calibration are not applied. Missing SPL
//! stays unknown. Dropped or nonfinite samples invalidate the affected frame.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use math_audio_dsp::analysis::SpectrumAnalyzer;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Explicit per-machine settings for input-only live monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveLevelConfig {
    /// Exact input device ID or a uniquely matching name.
    pub input_device: String,
    /// Zero-based hardware input channel.
    pub input_channel: u16,
    /// Required hardware rate; monitoring does not resample.
    pub sample_rate_hz: u32,
    /// Power-of-two nonoverlapping analysis block, from 256 through 16384 samples.
    pub fft_size: usize,
    /// Maximum monitoring duration, from zero exclusive through 3600 seconds.
    pub duration_secs: f64,
}

impl LiveLevelConfig {
    /// Validate storage, duration, channel and rate limits before opening devices.
    ///
    /// # Errors
    /// Rejects empty selectors, invalid rates/channels, FFT sizes or durations.
    pub fn validate(&self) -> Result<(), String> {
        if self.input_device.trim().is_empty()
            || self.input_channel >= 64
            || !(8000..=384000).contains(&self.sample_rate_hz)
            || !(256..=16384).contains(&self.fft_size)
            || !self.fft_size.is_power_of_two()
            || !self.duration_secs.is_finite()
            || self.duration_secs <= 0.0
            || self.duration_secs > 3600.0
        {
            return Err("live monitoring requires an explicit input, channel <64, rate 8000..384000, power-of-two FFT 256..16384 and duration (0,3600]".into());
        }
        Ok(())
    }
}

/// One raw analysis frame, with explicit gaps and units.
#[derive(Debug, Clone, Serialize)]
pub struct LiveLevelFrame {
    /// Frame contract version.
    pub version: u32,
    /// Monotonically increasing analysis frame number.
    pub sequence: u64,
    /// Elapsed wall time since starting the input stream.
    pub elapsed_secs: f64,
    /// Negotiated hardware sample rate.
    pub sample_rate_hz: u32,
    /// Actual analysis block size.
    pub fft_size: usize,
    /// Raw RMS level; digital silence and invalid frames have no finite dB value.
    pub rms_dbfs: Option<f64>,
    /// Raw sample peak, before windowing; invalid frames remain unknown.
    pub peak_dbfs: Option<f64>,
    /// Absolute SPL is unavailable in this raw monitor.
    pub level_spl_db: Option<f64>,
    /// True only when no input gap or nonfinite sample affects this frame.
    pub continuous: bool,
    /// Samples at or beyond digital full scale in this block.
    pub clipped_samples: usize,
    /// Nonfinite samples in this block.
    pub nonfinite_samples: usize,
    /// Cumulative input samples lost at the bounded capture ring.
    pub dropped_samples: u64,
    /// Spectral bin frequencies in Hz, absent for invalid frames.
    pub frequencies_hz: Vec<f32>,
    /// One-sided peak-amplitude dBFS, with symmetric Hann coherent-gain correction.
    pub spectrum_peak_dbfs: Vec<f32>,
}

/// Final live-stream accounting after the input device has been released.
#[derive(Debug, Clone, Serialize)]
pub struct LiveLevelSummary {
    /// Summary contract version.
    pub version: u32,
    /// Caller-selected input identity, independent of device ordering.
    pub requested_input_device: String,
    /// Zero-based hardware channel selected for this monitor.
    pub input_channel: u16,
    /// Exact requested and negotiated rate for an opened stream.
    pub sample_rate_hz: u32,
    /// Nonoverlapping analysis block size.
    pub fft_size: usize,
    /// Actual device ID when available.
    pub input_device_id: Option<String>,
    /// Negotiated sample format.
    pub input_sample_format: String,
    /// Emitted full analysis frames.
    pub frames: u64,
    /// Frames invalidated by drops or nonfinite values.
    pub invalid_frames: u64,
    /// Input samples delivered by the driver, including ring overruns.
    pub received_samples: u64,
    /// Samples not admitted to the bounded ring.
    pub dropped_samples: u64,
    /// Remaining partial analysis samples, excluded from frames.
    pub incomplete_samples: usize,
    /// Elapsed wall time at stop.
    pub elapsed_secs: f64,
    /// Observed emitted frame rate, including slow consumers.
    pub emitted_frames_per_second: f64,
    /// Whether the caller requested cancellation.
    pub cancelled: bool,
}

fn exact_device_index(identities: &[(String, String)], selector: &str) -> Result<usize, String> {
    let by_id: Vec<_> = identities
        .iter()
        .enumerate()
        .filter(|(_, (id, _))| id == selector)
        .map(|(index, _)| index)
        .collect();
    if by_id.len() == 1 {
        return Ok(by_id[0]);
    }
    if by_id.len() > 1 {
        return Err("ambiguous input device ID".into());
    }
    let by_name: Vec<_> = identities
        .iter()
        .enumerate()
        .filter(|(_, (_, name))| name == selector)
        .map(|(index, _)| index)
        .collect();
    match by_name.as_slice() {
        [index] => Ok(*index),
        [] => Err("input device must match an exact ID or unique full name".into()),
        _ => Err("input device name is ambiguous; select its exact ID".into()),
    }
}

fn level_db(amplitude: f64) -> Option<f64> {
    (amplitude > 0.0 && amplitude.is_finite()).then(|| 20.0 * amplitude.log10())
}

fn analyze_frame(
    analyzer: &mut SpectrumAnalyzer,
    samples: &[f32],
    config: &LiveLevelConfig,
    sequence: u64,
    elapsed_secs: f64,
    dropped_samples: u64,
    gap: bool,
) -> Result<LiveLevelFrame, String> {
    let nonfinite_samples = samples.iter().filter(|sample| !sample.is_finite()).count();
    let clipped_samples = samples
        .iter()
        .filter(|sample| sample.is_finite() && sample.abs() >= 1.0)
        .count();
    let continuous = !gap && nonfinite_samples == 0;
    let (rms_dbfs, peak_dbfs, frequencies_hz, spectrum_peak_dbfs) = if continuous {
        let mean_square = samples
            .iter()
            .map(|sample| f64::from(*sample).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        let peak = samples
            .iter()
            .map(|sample| f64::from(sample.abs()))
            .fold(0.0, f64::max);
        let (frequencies, spectrum, _) = analyzer.welch(samples, config.sample_rate_hz, 0.0)?;
        if spectrum.iter().any(|value| !value.is_finite()) {
            return Err("live spectrum estimator returned nonfinite values".into());
        }
        (
            level_db(mean_square.sqrt()),
            level_db(peak),
            frequencies,
            spectrum,
        )
    } else {
        (None, None, Vec::new(), Vec::new())
    };
    Ok(LiveLevelFrame {
        version: 1,
        sequence,
        elapsed_secs,
        sample_rate_hz: config.sample_rate_hz,
        fft_size: config.fft_size,
        rms_dbfs,
        peak_dbfs,
        level_spl_db: None,
        continuous,
        clipped_samples,
        nonfinite_samples,
        dropped_samples,
        frequencies_hz,
        spectrum_peak_dbfs,
    })
}

fn enqueue_input<T>(
    producer: &mut rtrb::Producer<(u64, f32)>,
    data: &[T],
    channels: usize,
    channel: usize,
    received: &AtomicU64,
    dropped: &AtomicU64,
) where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let first_index = received.fetch_add((data.len() / channels) as u64, Ordering::Relaxed);
    let mut lost = 0;
    for (offset, frame) in data.chunks_exact(channels).enumerate() {
        let sample = frame[channel].to_sample::<f32>();
        if producer
            .push((first_index + offset as u64, sample))
            .is_err()
        {
            lost += 1;
        }
    }
    dropped.fetch_add(lost, Ordering::Relaxed);
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channel: usize,
    mut producer: rtrb::Producer<(u64, f32)>,
    received: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    failed: Arc<AtomicBool>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = usize::from(config.channels);
    device.build_input_stream(
        config,
        move |data: &[T], _| {
            enqueue_input(&mut producer, data, channels, channel, &received, &dropped);
        },
        move |_| {
            failed.store(true, Ordering::Relaxed);
        },
        None,
    )
}

/// Monitor a selected input with bounded storage and cancellable device ownership.
///
/// The audio callback only converts and enqueues samples. Analysis and `emit`
/// execute on the caller thread. Slow consumers produce reported sample drops;
/// frames spanning drops are invalid. The stream is dropped on every exit.
/// At most four FFT blocks are queued; no full-session sample history is kept.
///
/// # Errors
/// Rejects invalid settings, unsupported formats/rates/channels, device or
/// analysis errors, and errors returned by the consumer. No output audio plays.
pub fn stream_live_levels(
    config: &LiveLevelConfig,
    cancel: &AtomicBool,
    mut emit: impl FnMut(&LiveLevelFrame) -> Result<(), String>,
) -> Result<LiveLevelSummary, String> {
    config.validate()?;
    if cancel.load(Ordering::Relaxed) {
        return Ok(LiveLevelSummary {
            version: 1,
            requested_input_device: config.input_device.clone(),
            input_channel: config.input_channel,
            sample_rate_hz: config.sample_rate_hz,
            fft_size: config.fft_size,
            input_device_id: None,
            input_sample_format: "not_opened".into(),
            frames: 0,
            invalid_frames: 0,
            received_samples: 0,
            dropped_samples: 0,
            incomplete_samples: 0,
            elapsed_secs: 0.0,
            emitted_frames_per_second: 0.0,
            cancelled: true,
        });
    }
    #[cfg(not(all(target_os = "windows", feature = "asio")))]
    if config.input_device.starts_with("ASIO:") {
        return Err("ASIO monitoring requires Windows and the asio feature".into());
    }
    let host = crate::devices::get_host_for_device(Some(&config.input_device));
    #[cfg(all(target_os = "windows", feature = "asio"))]
    if config.input_device.starts_with("ASIO:") && host.id() != cpal::HostId::Asio {
        return Err("requested ASIO host is unavailable".into());
    }
    let devices: Vec<_> = host
        .input_devices()
        .map_err(|error| format!("cannot enumerate input devices: {error}"))?
        .collect();
    let identities: Vec<_> = devices
        .iter()
        .map(|device| {
            (
                device
                    .id()
                    .ok()
                    .map(|id| id.to_string())
                    .unwrap_or_default(),
                device
                    .description()
                    .ok()
                    .map(|description| description.name().to_owned())
                    .unwrap_or_default(),
            )
        })
        .collect();
    let selector = config
        .input_device
        .strip_prefix("ASIO:")
        .unwrap_or(&config.input_device);
    let index = exact_device_index(&identities, selector)?;
    let device = &devices[index];
    let supported = device
        .supported_input_configs()
        .map_err(|error| format!("cannot inspect input formats: {error}"))?
        .filter(|range| {
            range.channels() > config.input_channel
                && range.min_sample_rate() <= config.sample_rate_hz
                && range.max_sample_rate() >= config.sample_rate_hz
        })
        .find(|range| {
            matches!(
                range.sample_format(),
                cpal::SampleFormat::F32
                    | cpal::SampleFormat::F64
                    | cpal::SampleFormat::I16
                    | cpal::SampleFormat::I32
                    | cpal::SampleFormat::U16
                    | cpal::SampleFormat::U32
            )
        })
        .ok_or(
            "input does not support the requested exact rate/channel and a supported sample format",
        )?
        .with_sample_rate(config.sample_rate_hz);
    let format = supported.sample_format();
    let device_id = device.id().ok().map(|id| id.to_string());
    let stream_config = supported.config();
    let (producer, mut consumer) = rtrb::RingBuffer::new(config.fft_size * 4);
    let received = Arc::new(AtomicU64::new(0));
    let dropped = Arc::new(AtomicU64::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    macro_rules! input_stream {
        ($ty:ty) => {
            build_stream::<$ty>(
                device,
                &stream_config,
                usize::from(config.input_channel),
                producer,
                Arc::clone(&received),
                Arc::clone(&dropped),
                Arc::clone(&failed),
            )
        };
    }
    let stream = match format {
        cpal::SampleFormat::F32 => input_stream!(f32),
        cpal::SampleFormat::F64 => input_stream!(f64),
        cpal::SampleFormat::I16 => input_stream!(i16),
        cpal::SampleFormat::I32 => input_stream!(i32),
        cpal::SampleFormat::U16 => input_stream!(u16),
        cpal::SampleFormat::U32 => input_stream!(u32),
        _ => return Err("unsupported input sample format".into()),
    }
    .map_err(|error| format!("cannot build live input stream: {error}"))?;
    let mut analyzer = SpectrumAnalyzer::new(config.fft_size);
    let mut samples = Vec::with_capacity(config.fft_size);
    let start = Instant::now();
    let limit = Duration::from_secs_f64(config.duration_secs);
    stream
        .play()
        .map_err(|error| format!("cannot start live input stream: {error}"))?;
    let mut sequence = 0;
    let mut invalid_frames = 0;
    let mut last_sample_index = None;
    let mut gap = false;
    while start.elapsed() < limit && !cancel.load(Ordering::Relaxed) {
        if failed.load(Ordering::Relaxed) {
            return Err("live input stream failed; device released".into());
        }
        while samples.len() < config.fft_size {
            match consumer.pop() {
                Ok((index, sample)) => {
                    if last_sample_index.is_some_and(|previous| index != previous + 1) {
                        gap = true;
                    }
                    last_sample_index = Some(index);
                    samples.push(sample);
                }
                Err(_) => break,
            }
        }
        if samples.len() == config.fft_size {
            let total_drops = dropped.load(Ordering::Relaxed);
            let frame = analyze_frame(
                &mut analyzer,
                &samples,
                config,
                sequence,
                start.elapsed().as_secs_f64(),
                total_drops,
                gap,
            )?;
            invalid_frames += u64::from(!frame.continuous);
            emit(&frame)?;
            sequence += 1;
            samples.clear();
            gap = false;
        } else {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    drop(stream);
    if failed.load(Ordering::Relaxed) {
        return Err("live input stream failed; device released".into());
    }
    let elapsed_secs = start.elapsed().as_secs_f64();
    Ok(LiveLevelSummary {
        version: 1,
        requested_input_device: config.input_device.clone(),
        input_channel: config.input_channel,
        sample_rate_hz: config.sample_rate_hz,
        fft_size: config.fft_size,
        input_device_id: device_id,
        input_sample_format: format.to_string(),
        frames: sequence,
        invalid_frames,
        received_samples: received.load(Ordering::Relaxed),
        dropped_samples: dropped.load(Ordering::Relaxed),
        incomplete_samples: samples.len() + consumer.slots(),
        elapsed_secs,
        emitted_frames_per_second: sequence as f64 / elapsed_secs,
        cancelled: cancel.load(Ordering::Relaxed),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> LiveLevelConfig {
        LiveLevelConfig {
            input_device: "explicit-input".into(),
            input_channel: 0,
            sample_rate_hz: 48000,
            fft_size: 4096,
            duration_secs: 1.0,
        }
    }

    #[test]
    fn independent_tone_matches_rms_and_peak_spectrum_units() {
        let config = config();
        let samples: Vec<_> = (0..config.fft_size)
            .map(|index| {
                (0.5 * (2.0 * std::f64::consts::PI * 64.0 * index as f64 / config.fft_size as f64)
                    .sin()) as f32
            })
            .collect();
        let mut analyzer = SpectrumAnalyzer::new(config.fft_size);
        let frame = analyze_frame(&mut analyzer, &samples, &config, 0, 0.1, 0, false).unwrap();
        assert!((frame.rms_dbfs.unwrap() - 20.0 * (0.5_f64 / 2.0_f64.sqrt()).log10()).abs() < 1e-6);
        assert!((frame.peak_dbfs.unwrap() - 20.0 * 0.5_f64.log10()).abs() < 1e-6);
        assert!((f64::from(frame.spectrum_peak_dbfs[64]) - 20.0 * 0.5_f64.log10()).abs() < 0.01);
        assert_eq!(frame.frequencies_hz[64], 750.0);
        assert!(frame.level_spl_db.is_none());
    }

    #[test]
    fn silence_invalid_samples_and_gaps_have_explicit_unknowns() {
        let config = config();
        let mut analyzer = SpectrumAnalyzer::new(config.fft_size);
        let mut samples = vec![0.0; config.fft_size];
        let silence = analyze_frame(&mut analyzer, &samples, &config, 0, 0.1, 0, false).unwrap();
        assert!(silence.continuous && silence.rms_dbfs.is_none() && silence.peak_dbfs.is_none());
        serde_json::to_vec(&silence).unwrap();
        samples[3] = f32::NAN;
        samples[4] = 1.1;
        let invalid = analyze_frame(&mut analyzer, &samples, &config, 1, 0.2, 0, false).unwrap();
        assert!(!invalid.continuous && invalid.spectrum_peak_dbfs.is_empty());
        assert_eq!(invalid.nonfinite_samples, 1);
        assert_eq!(invalid.clipped_samples, 1);
        samples[3] = 0.0;
        let gap = analyze_frame(&mut analyzer, &samples, &config, 2, 0.3, 10, true).unwrap();
        assert!(!gap.continuous && gap.rms_dbfs.is_none());
        assert_eq!(gap.dropped_samples, 10);
    }

    #[test]
    fn bounded_ring_preserves_gap_identity_after_older_samples_are_drained() {
        let (mut producer, mut consumer) = rtrb::RingBuffer::new(8);
        let received = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        enqueue_input(&mut producer, &[0.0_f32; 12], 1, 0, &received, &dropped);
        assert_eq!(received.load(Ordering::Relaxed), 12);
        assert_eq!(dropped.load(Ordering::Relaxed), 4);
        for expected in 0..7 {
            assert_eq!(consumer.pop().unwrap().0, expected);
        }
        enqueue_input(&mut producer, &[0.0_f32; 2], 1, 0, &received, &dropped);
        assert_eq!(consumer.pop().unwrap().0, 7);
        assert_eq!(consumer.pop().unwrap().0, 12);
        assert_eq!(consumer.pop().unwrap().0, 13);
        assert!(consumer.pop().is_err());
    }

    #[test]
    fn monitoring_selection_refuses_fuzzy_or_duplicate_names() {
        let identities = vec![
            ("id-1".into(), "USB microphone".into()),
            ("id-2".into(), "USB microphone".into()),
        ];
        assert_eq!(exact_device_index(&identities, "id-2").unwrap(), 1);
        assert!(exact_device_index(&identities, "USB microphone").is_err());
        assert!(exact_device_index(&identities, "USB").is_err());
        assert!(exact_device_index(&identities, "missing").is_err());
    }

    #[test]
    fn already_cancelled_monitor_never_opens_missing_device() {
        let config = config();
        let summary = stream_live_levels(&config, &AtomicBool::new(true), |_| {
            panic!("no frame expected")
        })
        .unwrap();
        assert!(summary.cancelled);
        assert_eq!(summary.frames, 0);
        assert_eq!(summary.input_sample_format, "not_opened");
    }

    #[test]
    fn invalid_monitor_settings_fail_before_device_access() {
        let mut config = config();
        config.input_device.clear();
        assert!(config.validate().is_err());
        config.input_device = "input".into();
        for size in [0, 255, 257, 32768] {
            config.fft_size = size;
            assert!(config.validate().is_err());
        }
        config.fft_size = 4096;
        for duration in [0.0, -1.0, f64::NAN, f64::INFINITY, 3601.0] {
            config.duration_secs = duration;
            assert!(config.validate().is_err());
        }
    }
}
