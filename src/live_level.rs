//! Bounded raw and optionally calibrated live level and RTA monitoring.
//!
//! Raw RMS and peak levels remain dBFS, and the compatibility spectrum remains
//! the existing symmetric-Hann peak-amplitude estimate. An explicit immutable
//! machine-bound profile can add one-sided Hann PSD, relative response correction,
//! and band-limited RMS pressure/SPL. No calibration is inferred from device names.
//! Gaps, nonfinite samples, clipping, and binding mismatches withhold calibrated data.

mod profile;
mod psd;

#[doc(inline)]
pub use profile::{
    LiveCalibrationBinding, LiveCalibrationProfile, LiveCalibrationProfileInput,
    LiveCalibrationStatus, LiveGainBasis, LiveLevelMachineSettings, LiveResponseCurveConvention,
    LiveResponseCurveFormat, LiveResponseCurveInput, LiveRmsSplAnchorInput, LiveSplWeighting,
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use math_audio_dsp::analysis::SpectrumAnalyzer;
use psd::{PsdAnalyzer, integrate_band_power};
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
    /// Legacy broadband SPL field; remains unset in every live mode.
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
    /// PSD frequency grid in Hz, including DC and Nyquist.
    pub psd_frequencies_hz: Vec<f64>,
    /// Raw one-sided symmetric-Hann PSD in normalized FS²/Hz.
    pub spectrum_psd_fs2_per_hz: Vec<f64>,
    /// Response-corrected one-sided PSD in FS²/Hz; uncovered bins are `None`.
    pub response_corrected_psd_fs2_per_hz: Option<Vec<Option<f64>>>,
    /// Absolute one-sided pressure PSD in Pa²/Hz; uncovered bins are `None`.
    pub pressure_psd_pa2_per_hz: Option<Vec<Option<f64>>>,
    /// Explicit integration band used for any reported RMS pressure and SPL.
    pub spl_band_hz: Option<[f64; 2]>,
    /// Band-limited unweighted RMS pressure in Pa.
    pub band_limited_rms_pressure_pa: Option<f64>,
    /// Band-limited unweighted SPL in dB re 20 µPa RMS.
    pub band_limited_spl_db: Option<f64>,
    /// Machine-readable availability or withholding reason for a requested SPL band.
    pub band_limited_spl_status: Option<LiveBandSplStatus>,
    /// Calibration validity for this frame, separate from raw frame continuity.
    pub calibration_frame_status: LiveCalibrationFrameStatus,
}

/// Availability of the requested absolute, band-limited SPL result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveBandSplStatus {
    /// No SPL band was requested.
    NotRequested,
    /// No validated RMS pressure anchor is present.
    NoAbsoluteAnchor,
    /// The input binding does not match the calibration profile.
    BindingMismatch,
    /// A sample gap invalidated the frame.
    InputGap,
    /// Nonfinite samples invalidated the frame.
    NonfiniteSamples,
    /// Clipped samples invalidated the frame.
    ClippedSamples,
    /// At least one frequency cell overlapping the requested band lacks calibration coverage.
    ResponseCoverageInsufficient,
    /// The calibrated band contained digital zero, so SPL has no finite logarithm.
    DigitalSilence,
    /// An RMS pressure and finite band-limited SPL are available.
    Available,
}

/// Per-frame reason calibrated values are present or withheld.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveCalibrationFrameStatus {
    /// No live calibration profile was requested.
    NotConfigured,
    /// The profile did not match the opened machine input path.
    BindingMismatch,
    /// Input samples were lost before analysis.
    InputGap,
    /// One or more input samples were nonfinite.
    NonfiniteSamples,
    /// One or more input samples reached or exceeded full scale.
    ClippedSamples,
    /// The frame passed the configured calibration validity checks.
    Valid,
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
    /// Machine binding result for the requested live calibration profile.
    pub calibration_status: LiveCalibrationStatus,
    /// CPAL host API used by the opened input, when a device was opened.
    pub input_host_api: Option<String>,
    /// Expected immutable profile binding, when a profile was supplied.
    pub calibration_profile_binding: Option<LiveCalibrationBinding>,
    /// Digest of the exact retained response bytes, when a curve was supplied.
    pub calibration_response_sha256: Option<String>,
    /// Explicit parser selected for the retained response bytes.
    pub calibration_response_format: Option<LiveResponseCurveFormat>,
    /// Response convention; positive dB means the microphone is too loud.
    pub calibration_response_convention: Option<LiveResponseCurveConvention>,
    /// Frequency where response deviation is normalized to 0 dB, in Hz.
    pub calibration_response_reference_frequency_hz: Option<f64>,
    /// Operator-provided machine settings, including gain attestation.
    pub machine_settings: Option<LiveLevelMachineSettings>,
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

#[derive(Clone, Copy)]
struct LiveFrameContext<'a> {
    config: &'a LiveLevelConfig,
    sequence: u64,
    elapsed_secs: f64,
    dropped_samples: u64,
    gap: bool,
    calibration: Option<(&'a LiveCalibrationProfile, &'a LiveLevelMachineSettings)>,
    calibration_status: LiveCalibrationStatus,
}

fn calibration_requested_for_frame<'a>(
    machine_settings: Option<&'a LiveLevelMachineSettings>,
    profile: Option<&'a LiveCalibrationProfile>,
) -> Option<(&'a LiveCalibrationProfile, &'a LiveLevelMachineSettings)> {
    profile.zip(machine_settings)
}

impl<'a> LiveFrameContext<'a> {
    fn new(
        config: &'a LiveLevelConfig,
        sequence: u64,
        elapsed_secs: f64,
        dropped_samples: u64,
        gap: bool,
        calibration: Option<(&'a LiveCalibrationProfile, &'a LiveLevelMachineSettings)>,
        calibration_status: LiveCalibrationStatus,
    ) -> Self {
        Self {
            config,
            sequence,
            elapsed_secs,
            dropped_samples,
            gap,
            calibration,
            calibration_status,
        }
    }
}

fn analyze_frame_with_calibration(
    analyzer: &mut SpectrumAnalyzer,
    psd_analyzer: &mut PsdAnalyzer,
    samples: &[f32],
    context: LiveFrameContext<'_>,
) -> Result<LiveLevelFrame, String> {
    let LiveFrameContext {
        config,
        sequence,
        elapsed_secs,
        dropped_samples,
        gap,
        calibration,
        calibration_status,
    } = context;
    let nonfinite_samples = samples.iter().filter(|sample| !sample.is_finite()).count();
    let clipped_samples = samples
        .iter()
        .filter(|sample| sample.is_finite() && sample.abs() >= 1.0)
        .count();
    let continuous = !gap && nonfinite_samples == 0;
    let (
        rms_dbfs,
        peak_dbfs,
        frequencies_hz,
        spectrum_peak_dbfs,
        psd_frequencies_hz,
        spectrum_psd_fs2_per_hz,
    ) = if continuous {
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
        psd_analyzer
            .analyze(samples, config.sample_rate_hz)
            .map_err(str::to_owned)?;
        (
            level_db(mean_square.sqrt()),
            level_db(peak),
            frequencies,
            spectrum,
            psd_analyzer.frequencies_hz().to_vec(),
            psd_analyzer.psd_fs2_per_hz().to_vec(),
        )
    } else {
        (None, None, Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };

    let frame_is_calibratable = continuous && clipped_samples == 0;
    let calibration_matches = matches!(
        calibration_status,
        LiveCalibrationStatus::RelativeOnly | LiveCalibrationStatus::Matched { .. }
    );
    let (response_corrected_psd_fs2_per_hz, pressure_psd_pa2_per_hz) =
        if frame_is_calibratable && calibration_matches {
            calibration.map_or((None, None), |(profile, _)| {
                (
                    profile.response_corrected_psd(&psd_frequencies_hz, &spectrum_psd_fs2_per_hz),
                    profile.pressure_psd(&psd_frequencies_hz, &spectrum_psd_fs2_per_hz),
                )
            })
        } else {
            (None, None)
        };
    let (spl_band_hz, band_limited_rms_pressure_pa, band_limited_spl_db, band_limited_spl_status) =
        match calibration {
            Some((profile, machine_settings)) if machine_settings.spl_band_hz.is_some() => {
                let band_hz = machine_settings.spl_band_hz;
                let status = match calibration_status {
                    LiveCalibrationStatus::NotConfigured
                    | LiveCalibrationStatus::NotChecked
                    | LiveCalibrationStatus::Mismatch(_) => {
                        Some(LiveBandSplStatus::BindingMismatch)
                    }
                    LiveCalibrationStatus::RelativeOnly | LiveCalibrationStatus::Matched { .. }
                        if !frame_is_calibratable =>
                    {
                        Some(if gap {
                            LiveBandSplStatus::InputGap
                        } else if nonfinite_samples > 0 {
                            LiveBandSplStatus::NonfiniteSamples
                        } else {
                            LiveBandSplStatus::ClippedSamples
                        })
                    }
                    LiveCalibrationStatus::RelativeOnly | LiveCalibrationStatus::Matched { .. }
                        if !profile.has_spl_anchor() =>
                    {
                        Some(LiveBandSplStatus::NoAbsoluteAnchor)
                    }
                    LiveCalibrationStatus::RelativeOnly | LiveCalibrationStatus::Matched { .. } => {
                        None
                    }
                };
                if let Some(status) = status {
                    (band_hz, None, None, Some(status))
                } else {
                    let band_power = pressure_psd_pa2_per_hz.as_deref().zip(band_hz).and_then(
                        |(spectrum, band)| {
                            integrate_band_power(
                                spectrum,
                                config.fft_size,
                                config.sample_rate_hz,
                                band,
                            )
                        },
                    );
                    match band_power {
                        None => (
                            band_hz,
                            None,
                            None,
                            Some(LiveBandSplStatus::ResponseCoverageInsufficient),
                        ),
                        Some(0.0) => (
                            band_hz,
                            Some(0.0),
                            None,
                            Some(LiveBandSplStatus::DigitalSilence),
                        ),
                        Some(power) => {
                            let pressure = power.sqrt();
                            let spl = 20.0 * (pressure / 20.0e-6).log10();
                            (
                                band_hz,
                                Some(pressure),
                                Some(spl),
                                Some(LiveBandSplStatus::Available),
                            )
                        }
                    }
                }
            }
            Some((_, machine_settings)) => (
                machine_settings.spl_band_hz,
                None,
                None,
                Some(LiveBandSplStatus::NotRequested),
            ),
            None => (None, None, None, None),
        };
    let calibration_frame_status = match calibration_status {
        LiveCalibrationStatus::NotConfigured => LiveCalibrationFrameStatus::NotConfigured,
        LiveCalibrationStatus::NotChecked | LiveCalibrationStatus::Mismatch(_) => {
            LiveCalibrationFrameStatus::BindingMismatch
        }
        LiveCalibrationStatus::RelativeOnly | LiveCalibrationStatus::Matched { .. } => {
            if gap {
                LiveCalibrationFrameStatus::InputGap
            } else if nonfinite_samples > 0 {
                LiveCalibrationFrameStatus::NonfiniteSamples
            } else if clipped_samples > 0 {
                LiveCalibrationFrameStatus::ClippedSamples
            } else {
                LiveCalibrationFrameStatus::Valid
            }
        }
    };
    Ok(LiveLevelFrame {
        version: 2,
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
        psd_frequencies_hz,
        spectrum_psd_fs2_per_hz,
        response_corrected_psd_fs2_per_hz,
        pressure_psd_pa2_per_hz,
        spl_band_hz,
        band_limited_rms_pressure_pa,
        band_limited_spl_db,
        band_limited_spl_status,
        calibration_frame_status,
    })
}

#[cfg(test)]
fn analyze_frame(
    analyzer: &mut SpectrumAnalyzer,
    samples: &[f32],
    config: &LiveLevelConfig,
    sequence: u64,
    elapsed_secs: f64,
    dropped_samples: u64,
    gap: bool,
) -> Result<LiveLevelFrame, String> {
    let mut psd_analyzer = PsdAnalyzer::new(config.fft_size).map_err(str::to_owned)?;
    analyze_frame_with_calibration(
        analyzer,
        &mut psd_analyzer,
        samples,
        LiveFrameContext::new(
            config,
            sequence,
            elapsed_secs,
            dropped_samples,
            gap,
            None,
            LiveCalibrationStatus::NotConfigured,
        ),
    )
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

/// Monitors a selected input as raw digital levels with bounded storage.
///
/// The audio callback only converts and enqueues samples. Analysis and `emit`
/// execute on the caller thread. Slow consumers produce reported sample drops;
/// frames spanning drops are invalid. The stream is dropped on every exit.
/// At most four FFT blocks are queued; no full-session sample history is kept.
/// Use [`stream_live_levels_calibrated`] to add an explicitly bound profile.
///
/// # Errors
/// Rejects invalid settings, unsupported formats/rates/channels, device or
/// analysis errors, and errors returned by the consumer. No output audio plays.
pub fn stream_live_levels(
    config: &LiveLevelConfig,
    cancel: &AtomicBool,
    emit: impl FnMut(&LiveLevelFrame) -> Result<(), String>,
) -> Result<LiveLevelSummary, String> {
    stream_live_levels_inner(config, None, None, cancel, emit)
}

/// Monitors a selected input and applies a validated, machine-bound profile.
///
/// The profile only affects calibrated PSD and pressure fields. Raw dBFS fields
/// remain present when the opened route mismatches the profile; calibrated values
/// are then withheld and the mismatch is reported in each frame and the summary.
/// CPAL cannot read hardware gain back, so the explicit machine setting records a
/// user attestation that gain was checked and will remain fixed.
///
/// # Errors
/// Rejects invalid raw settings, invalid machine declarations, unsupported hardware,
/// analysis failures, and consumer errors. Device/profile mismatches retain raw data.
pub fn stream_live_levels_calibrated(
    config: &LiveLevelConfig,
    machine_settings: &LiveLevelMachineSettings,
    profile: &LiveCalibrationProfile,
    cancel: &AtomicBool,
    emit: impl FnMut(&LiveLevelFrame) -> Result<(), String>,
) -> Result<LiveLevelSummary, String> {
    config.validate()?;
    machine_settings.validate()?;
    if machine_settings
        .spl_band_hz
        .is_some_and(|band| band[1] > f64::from(config.sample_rate_hz) / 2.0)
    {
        return Err("live SPL band must not exceed the configured Nyquist frequency".into());
    }
    stream_live_levels_inner(config, Some(machine_settings), Some(profile), cancel, emit)
}

fn stream_live_levels_inner(
    config: &LiveLevelConfig,
    machine_settings: Option<&LiveLevelMachineSettings>,
    profile: Option<&LiveCalibrationProfile>,
    cancel: &AtomicBool,
    mut emit: impl FnMut(&LiveLevelFrame) -> Result<(), String>,
) -> Result<LiveLevelSummary, String> {
    config.validate()?;
    let initial_calibration_status = if profile.is_some() {
        LiveCalibrationStatus::NotChecked
    } else {
        LiveCalibrationStatus::NotConfigured
    };
    if cancel.load(Ordering::Relaxed) {
        return Ok(LiveLevelSummary {
            version: 2,
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
            calibration_status: initial_calibration_status,
            input_host_api: None,
            calibration_profile_binding: profile.map(|profile| profile.binding().clone()),
            calibration_response_sha256: profile
                .and_then(LiveCalibrationProfile::response_sha256)
                .map(str::to_owned),
            calibration_response_format: profile.and_then(LiveCalibrationProfile::response_format),
            calibration_response_convention: profile
                .and_then(LiveCalibrationProfile::response_convention),
            calibration_response_reference_frequency_hz: profile
                .and_then(LiveCalibrationProfile::response_reference_frequency_hz),
            machine_settings: machine_settings.cloned(),
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
    let host_api = host.id().name().to_owned();
    let calibration_status = match (machine_settings, profile) {
        (Some(machine_settings), Some(profile)) => {
            let runtime_binding = LiveCalibrationBinding {
                microphone_id: machine_settings.microphone_id.clone(),
                host_api: host_api.clone(),
                input_device_id: device_id.clone().unwrap_or_default(),
                input_channel: config.input_channel,
                sample_rate_hz: stream_config.sample_rate,
                input_sample_format: format.to_string(),
                declared_input_gain_db: machine_settings.declared_input_gain_db,
                gain_attested: machine_settings.gain_attested,
                orientation: machine_settings.orientation,
            };
            match profile.match_binding(&runtime_binding) {
                Ok(()) if profile.has_response_curve() && !profile.has_spl_anchor() => {
                    LiveCalibrationStatus::RelativeOnly
                }
                Ok(()) => LiveCalibrationStatus::Matched {
                    response_curve: profile.has_response_curve(),
                    spl_anchor: profile.has_spl_anchor(),
                    gain_basis: LiveGainBasis::UserDeclared,
                },
                Err(reason) => LiveCalibrationStatus::Mismatch(reason),
            }
        }
        _ => LiveCalibrationStatus::NotConfigured,
    };
    let active_calibration = calibration_requested_for_frame(machine_settings, profile);
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
    let mut psd_analyzer = PsdAnalyzer::new(config.fft_size).map_err(str::to_owned)?;
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
            let frame = analyze_frame_with_calibration(
                &mut analyzer,
                &mut psd_analyzer,
                &samples,
                LiveFrameContext::new(
                    config,
                    sequence,
                    start.elapsed().as_secs_f64(),
                    total_drops,
                    gap,
                    active_calibration,
                    calibration_status,
                ),
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
        version: 2,
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
        calibration_status,
        input_host_api: Some(host_api),
        calibration_profile_binding: profile.map(|profile| profile.binding().clone()),
        calibration_response_sha256: profile
            .and_then(LiveCalibrationProfile::response_sha256)
            .map(str::to_owned),
        calibration_response_format: profile.and_then(LiveCalibrationProfile::response_format),
        calibration_response_convention: profile
            .and_then(LiveCalibrationProfile::response_convention),
        calibration_response_reference_frequency_hz: profile
            .and_then(LiveCalibrationProfile::response_reference_frequency_hz),
        machine_settings: machine_settings.cloned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn config() -> LiveLevelConfig {
        LiveLevelConfig {
            input_device: "explicit-input".into(),
            input_channel: 0,
            sample_rate_hz: 48000,
            fft_size: 4096,
            duration_secs: 1.0,
        }
    }

    fn binding() -> LiveCalibrationBinding {
        LiveCalibrationBinding {
            microphone_id: "room-mic-a".into(),
            host_api: "CoreAudio".into(),
            input_device_id: "coreaudio:input:42".into(),
            input_channel: 0,
            sample_rate_hz: 48_000,
            input_sample_format: "F32".into(),
            declared_input_gain_db: 12.0,
            gain_attested: true,
            orientation: crate::capture_session::CalibrationOrientation::OnAxis,
        }
    }

    fn response_bytes() -> &'static [u8] {
        b"20 0\n750 0\n1500 6\n24000 6\n"
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn calibration_profile(with_anchor: bool) -> LiveCalibrationProfile {
        calibration_profile_with_response(response_bytes(), with_anchor)
    }

    fn calibration_profile_with_response(
        response_bytes: &[u8],
        with_anchor: bool,
    ) -> LiveCalibrationProfile {
        let binding = binding();
        let response_curve = LiveResponseCurveInput {
            format: LiveResponseCurveFormat::Text,
            convention: LiveResponseCurveConvention::PositiveDbMeansMicrophoneTooLoud,
            reference_frequency_hz: 750.0,
            bytes: response_bytes.to_vec(),
            expected_sha256: Some(sha256_hex(response_bytes)),
        };
        let spl_anchor = with_anchor.then(|| LiveRmsSplAnchorInput {
            binding: binding.clone(),
            response_sha256: Some(sha256_hex(response_bytes)),
            reference_rms_full_scale: 0.1,
            reference_level_db_spl: 94.0,
            reference_frequency_hz: 750.0,
            band_hz: [100.0, 20_000.0],
            weighting: LiveSplWeighting::Z,
        });
        LiveCalibrationProfile::new(LiveCalibrationProfileInput {
            binding,
            response_curve: Some(response_curve),
            spl_anchor,
        })
        .unwrap()
    }

    fn absolute_anchor_only_profile() -> LiveCalibrationProfile {
        let binding = binding();
        LiveCalibrationProfile::new(LiveCalibrationProfileInput {
            binding: binding.clone(),
            response_curve: None,
            spl_anchor: Some(LiveRmsSplAnchorInput {
                binding,
                response_sha256: None,
                reference_rms_full_scale: 0.1,
                reference_level_db_spl: 94.0,
                reference_frequency_hz: 750.0,
                band_hz: [725.0, 775.0],
                weighting: LiveSplWeighting::Z,
            }),
        })
        .unwrap()
    }

    fn machine_settings() -> LiveLevelMachineSettings {
        LiveLevelMachineSettings {
            microphone_id: "room-mic-a".into(),
            declared_input_gain_db: 12.0,
            gain_attested: true,
            orientation: crate::capture_session::CalibrationOrientation::OnAxis,
            spl_band_hz: Some([100.0, 20_000.0]),
        }
    }

    fn matched_status(response_curve: bool, spl_anchor: bool) -> LiveCalibrationStatus {
        LiveCalibrationStatus::Matched {
            response_curve,
            spl_anchor,
            gain_basis: LiveGainBasis::UserDeclared,
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
        assert_eq!(frame.psd_frequencies_hz.len(), config.fft_size / 2 + 1);
        assert_eq!(frame.psd_frequencies_hz.last().copied(), Some(24_000.0));
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

    #[test]
    fn calibrated_frame_keeps_raw_units_and_reports_band_limited_spl() {
        let config = config();
        let samples: Vec<_> = (0..config.fft_size)
            .map(|index| {
                (0.2 * (std::f64::consts::TAU * 64.0 * index as f64 / config.fft_size as f64).sin())
                    as f32
            })
            .collect();
        let profile = calibration_profile(true);
        let machine = machine_settings();
        let mut analyzer = SpectrumAnalyzer::new(config.fft_size);
        let mut psd_analyzer = PsdAnalyzer::new(config.fft_size).unwrap();
        let frame = analyze_frame_with_calibration(
            &mut analyzer,
            &mut psd_analyzer,
            &samples,
            LiveFrameContext::new(
                &config,
                0,
                0.1,
                0,
                false,
                Some((&profile, &machine)),
                matched_status(true, true),
            ),
        )
        .unwrap();

        assert!((f64::from(frame.spectrum_peak_dbfs[64]) - 20.0 * 0.2_f64.log10()).abs() < 0.01);
        assert_eq!(frame.frequencies_hz[64], 750.0);
        assert_eq!(frame.psd_frequencies_hz[64], 750.0);
        assert_eq!(frame.version, 2);
        assert_eq!(
            frame.calibration_frame_status,
            LiveCalibrationFrameStatus::Valid
        );
        assert_eq!(frame.spl_band_hz, Some([100.0, 20_000.0]));
        assert!(frame.pressure_psd_pa2_per_hz.as_ref().unwrap()[64].is_some());
        assert!(frame.response_corrected_psd_fs2_per_hz.is_some());
        assert!(frame.band_limited_rms_pressure_pa.is_some());
        assert!(frame.band_limited_spl_db.is_some());
        assert_eq!(
            frame.band_limited_spl_status,
            Some(LiveBandSplStatus::Available)
        );
        assert!(frame.level_spl_db.is_none());
    }

    #[test]
    fn anchor_only_frame_reports_absolute_spl_and_tracks_amplitude_ratio() {
        let config = config();
        let samples: Vec<_> = (0..config.fft_size)
            .map(|index| {
                (0.2 * (std::f64::consts::TAU * 64.0 * index as f64 / config.fft_size as f64).sin())
                    as f32
            })
            .collect();
        let doubled_samples: Vec<_> = samples.iter().map(|sample| sample * 2.0).collect();
        let profile = absolute_anchor_only_profile();
        let mut machine = machine_settings();
        machine.spl_band_hz = Some([725.0, 775.0]);
        let mut analyzer = SpectrumAnalyzer::new(config.fft_size);
        let mut psd_analyzer = PsdAnalyzer::new(config.fft_size).unwrap();
        let analyze = |samples: &[f32], analyzer: &mut SpectrumAnalyzer, psd: &mut PsdAnalyzer| {
            analyze_frame_with_calibration(
                analyzer,
                psd,
                samples,
                LiveFrameContext::new(
                    &config,
                    0,
                    0.1,
                    0,
                    false,
                    Some((&profile, &machine)),
                    matched_status(false, true),
                ),
            )
            .unwrap()
        };
        let base = analyze(&samples, &mut analyzer, &mut psd_analyzer);
        let doubled = analyze(&doubled_samples, &mut analyzer, &mut psd_analyzer);

        let expected_spl_db = 94.0 + 20.0 * ((0.2 / 2.0_f64.sqrt()) / 0.1).log10();
        let reference_pressure_pa = 20.0e-6 * 10.0_f64.powf(94.0 / 20.0);
        let expected_rms_pressure_pa = reference_pressure_pa / 0.1 * (0.2 / 2.0_f64.sqrt());
        assert!(
            (base.band_limited_spl_db.unwrap() - expected_spl_db).abs() < 0.01,
            "measured {:?}, expected {expected_spl_db}",
            base.band_limited_spl_db
        );
        assert!(
            (base.band_limited_rms_pressure_pa.unwrap() - expected_rms_pressure_pa).abs() < 1e-5,
            "measured {:?}, expected {expected_rms_pressure_pa}",
            base.band_limited_rms_pressure_pa
        );
        assert!(
            (doubled.band_limited_rms_pressure_pa.unwrap()
                / base.band_limited_rms_pressure_pa.unwrap()
                - 2.0)
                .abs()
                < 1e-5
        );
        assert!(
            (doubled.band_limited_spl_db.unwrap()
                - base.band_limited_spl_db.unwrap()
                - 20.0 * 2.0_f64.log10())
            .abs()
                < 0.01
        );
    }

    #[test]
    fn response_only_profile_never_emits_pressure_or_absolute_spl() {
        let config = config();
        let samples = vec![0.1; config.fft_size];
        let profile = calibration_profile(false);
        let machine = machine_settings();
        let mut analyzer = SpectrumAnalyzer::new(config.fft_size);
        let mut psd_analyzer = PsdAnalyzer::new(config.fft_size).unwrap();
        let frame = analyze_frame_with_calibration(
            &mut analyzer,
            &mut psd_analyzer,
            &samples,
            LiveFrameContext::new(
                &config,
                0,
                0.1,
                0,
                false,
                Some((&profile, &machine)),
                LiveCalibrationStatus::RelativeOnly,
            ),
        )
        .unwrap();

        assert!(frame.response_corrected_psd_fs2_per_hz.is_some());
        assert!(frame.pressure_psd_pa2_per_hz.is_none());
        assert!(frame.band_limited_rms_pressure_pa.is_none());
        assert!(frame.band_limited_spl_db.is_none());
        assert_eq!(
            frame.band_limited_spl_status,
            Some(LiveBandSplStatus::NoAbsoluteAnchor)
        );
        assert!(frame.level_spl_db.is_none());
    }

    #[test]
    fn requested_band_reports_coverage_and_digital_silence_reasons() {
        let config = config();
        let narrow_profile = calibration_profile_with_response(b"700 0\n24000 0\n", true);
        let machine = machine_settings();
        let mut analyzer = SpectrumAnalyzer::new(config.fft_size);
        let mut psd_analyzer = PsdAnalyzer::new(config.fft_size).unwrap();
        let nonzero = vec![0.1; config.fft_size];
        let uncovered = analyze_frame_with_calibration(
            &mut analyzer,
            &mut psd_analyzer,
            &nonzero,
            LiveFrameContext::new(
                &config,
                0,
                0.1,
                0,
                false,
                Some((&narrow_profile, &machine)),
                matched_status(true, true),
            ),
        )
        .unwrap();
        assert_eq!(
            uncovered.band_limited_spl_status,
            Some(LiveBandSplStatus::ResponseCoverageInsufficient)
        );
        assert!(uncovered.band_limited_spl_db.is_none());

        let full_profile = calibration_profile(true);
        let silence = vec![0.0; config.fft_size];
        let silent = analyze_frame_with_calibration(
            &mut analyzer,
            &mut psd_analyzer,
            &silence,
            LiveFrameContext::new(
                &config,
                1,
                0.2,
                0,
                false,
                Some((&full_profile, &machine)),
                matched_status(true, true),
            ),
        )
        .unwrap();
        assert_eq!(
            silent.band_limited_spl_status,
            Some(LiveBandSplStatus::DigitalSilence)
        );
        assert_eq!(silent.band_limited_rms_pressure_pa, Some(0.0));
        assert!(silent.band_limited_spl_db.is_none());
    }

    #[test]
    fn cancelled_summary_retains_response_interpretation_metadata() {
        let config = config();
        let machine = machine_settings();
        let profile = calibration_profile(false);
        let summary = stream_live_levels_calibrated(
            &config,
            &machine,
            &profile,
            &AtomicBool::new(true),
            |_| panic!("no frame expected"),
        )
        .unwrap();

        assert_eq!(
            summary.calibration_response_format,
            Some(LiveResponseCurveFormat::Text)
        );
        assert_eq!(
            summary.calibration_response_convention,
            Some(LiveResponseCurveConvention::PositiveDbMeansMicrophoneTooLoud)
        );
        assert_eq!(
            summary.calibration_response_reference_frequency_hz,
            Some(750.0)
        );
        assert_eq!(summary.version, 2);
        assert!(summary.calibration_profile_binding.unwrap().gain_attested);
    }

    #[test]
    fn calibrated_values_are_withheld_for_mismatch_clipping_gaps_and_nonfinite_input() {
        let config = config();
        let profile = calibration_profile(true);
        let machine = machine_settings();
        let mut analyzer = SpectrumAnalyzer::new(config.fft_size);
        let mut psd_analyzer = PsdAnalyzer::new(config.fft_size).unwrap();
        let clean_samples = vec![0.1; config.fft_size];
        let mut mismatched_binding = profile.binding().clone();
        mismatched_binding.input_device_id.push_str("-different");
        let mismatch_status = LiveCalibrationStatus::Mismatch(
            profile.match_binding(&mismatched_binding).unwrap_err(),
        );
        let requested_calibration = calibration_requested_for_frame(Some(&machine), Some(&profile));
        assert!(requested_calibration.is_some());
        let mismatch = analyze_frame_with_calibration(
            &mut analyzer,
            &mut psd_analyzer,
            &clean_samples,
            LiveFrameContext::new(
                &config,
                0,
                0.1,
                0,
                false,
                requested_calibration,
                mismatch_status,
            ),
        )
        .unwrap();
        assert_eq!(
            mismatch.calibration_frame_status,
            LiveCalibrationFrameStatus::BindingMismatch
        );
        assert_eq!(mismatch.spl_band_hz, Some([100.0, 20_000.0]));
        assert!(mismatch.pressure_psd_pa2_per_hz.is_none());
        assert!(mismatch.response_corrected_psd_fs2_per_hz.is_none());
        assert!(mismatch.band_limited_spl_db.is_none());
        assert!(mismatch.band_limited_rms_pressure_pa.is_none());
        assert_eq!(
            mismatch.band_limited_spl_status,
            Some(LiveBandSplStatus::BindingMismatch)
        );

        let mut clipped_samples = clean_samples.clone();
        clipped_samples[20] = 1.0;
        let clipped = analyze_frame_with_calibration(
            &mut analyzer,
            &mut psd_analyzer,
            &clipped_samples,
            LiveFrameContext::new(
                &config,
                1,
                0.2,
                0,
                false,
                Some((&profile, &machine)),
                matched_status(true, true),
            ),
        )
        .unwrap();
        assert_eq!(
            clipped.calibration_frame_status,
            LiveCalibrationFrameStatus::ClippedSamples
        );
        assert!(clipped.pressure_psd_pa2_per_hz.is_none());
        assert!(clipped.band_limited_spl_db.is_none());
        assert_eq!(
            clipped.band_limited_spl_status,
            Some(LiveBandSplStatus::ClippedSamples)
        );

        let gap = analyze_frame_with_calibration(
            &mut analyzer,
            &mut psd_analyzer,
            &clean_samples,
            LiveFrameContext::new(
                &config,
                2,
                0.3,
                8,
                true,
                Some((&profile, &machine)),
                matched_status(true, true),
            ),
        )
        .unwrap();
        assert_eq!(
            gap.calibration_frame_status,
            LiveCalibrationFrameStatus::InputGap
        );
        assert!(gap.pressure_psd_pa2_per_hz.is_none());
        assert_eq!(
            gap.band_limited_spl_status,
            Some(LiveBandSplStatus::InputGap)
        );

        let mut nonfinite_samples = clean_samples;
        nonfinite_samples[21] = f32::INFINITY;
        let nonfinite = analyze_frame_with_calibration(
            &mut analyzer,
            &mut psd_analyzer,
            &nonfinite_samples,
            LiveFrameContext::new(
                &config,
                3,
                0.4,
                0,
                false,
                Some((&profile, &machine)),
                matched_status(true, true),
            ),
        )
        .unwrap();
        assert_eq!(
            nonfinite.calibration_frame_status,
            LiveCalibrationFrameStatus::NonfiniteSamples
        );
        assert!(nonfinite.pressure_psd_pa2_per_hz.is_none());
        assert_eq!(
            nonfinite.band_limited_spl_status,
            Some(LiveBandSplStatus::NonfiniteSamples)
        );
    }

    #[test]
    fn invalid_frame_does_not_poison_the_next_calibrated_frame() {
        let config = config();
        let profile = calibration_profile(true);
        let machine = machine_settings();
        let mut analyzer = SpectrumAnalyzer::new(config.fft_size);
        let mut psd_analyzer = PsdAnalyzer::new(config.fft_size).unwrap();
        let mut invalid_samples = vec![0.1; config.fft_size];
        invalid_samples[0] = f32::NAN;
        let invalid = analyze_frame_with_calibration(
            &mut analyzer,
            &mut psd_analyzer,
            &invalid_samples,
            LiveFrameContext::new(
                &config,
                0,
                0.1,
                0,
                false,
                Some((&profile, &machine)),
                matched_status(true, true),
            ),
        )
        .unwrap();
        assert!(invalid.pressure_psd_pa2_per_hz.is_none());

        let valid = analyze_frame_with_calibration(
            &mut analyzer,
            &mut psd_analyzer,
            &vec![0.1; config.fft_size],
            LiveFrameContext::new(
                &config,
                1,
                0.2,
                0,
                false,
                Some((&profile, &machine)),
                matched_status(true, true),
            ),
        )
        .unwrap();
        assert!(valid.pressure_psd_pa2_per_hz.is_some());
        assert_eq!(
            valid.calibration_frame_status,
            LiveCalibrationFrameStatus::Valid
        );
    }
}
