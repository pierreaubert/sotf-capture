//! Source-to-microphone ranges from sweep arrivals on raw takes.
//!
//! This offline stage runs inside `plan-process` after clock correction.
//! For every source/microphone pair it estimates the sweep arrival on the
//! original device-clock recording, converts the arrival into an absolute
//! distance when a time origin exists (loopback or surveyed emitter) or
//! into a relative delay otherwise, and persists the matrix with its
//! evidence. Multilateration then solves one 3D speaker position per
//! source row from the absolute ranges, gated by geometry rank and a
//! post-fit residual check (Q7).
//!
//! Arrival estimation reuses the envelope matched filter from the probe
//! path: [`cross_correlate_envelope`](math_audio_dsp::analysis::cross_correlate_envelope)
//! finds the strongest arrival and
//! [`pick_direct_arrival_from_envelope`](crate::signal_recorder::pick_direct_arrival_from_envelope)
//! finds the first significant one. A cell whose strongest arrival is not
//! its first is refused as `arrival_ambiguous` instead of reporting a
//! reflection as a distance.
//!
//! Single-interface assumption (localization.md §3): all microphones share
//! one ADC clock, so takes recorded in the same capture call share one
//! startup offset and one skew. Differential measurements inside such a
//! group (microphone minus loopback, microphone minus microphone) cancel
//! the shared terms without resampling.

// Rust guideline compliant 2026-02-21

use super::clock::io::{CLOCK_SKEW_SPREAD_PPM, ClockProcessedManifest};
use super::clock::{CaptureClockBasis, CaptureClockFit};
use super::{CaptureMicrophone, CaptureSessionPlan, CaptureSource};
use crate::quality::COHERENT_CAPTURE_SNR_THRESHOLD_DB;
use math_audio_dsp::analysis::cross_correlate_envelope;
use math_audio_dsp::capture_localization::{MIN_RANGES_FOR_FIX, geometry_rank, solve_position};
use math_audio_dsp::capture_tdoa::MIN_TDOA_CONFIDENCE_DB;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Write as _};
use std::path::Path;

/// Maximum tolerated disagreement between the direct and strongest sweep
/// arrival in one take, in samples (Q3).
///
/// Both estimators read the same envelope correlation: the direct picker
/// finds the first significant peak, the correlator the global maximum.
/// Beyond one sample the strongest arrival is a reflection, not the
/// direct sound, and the cell is refused instead of ranged.
const ARRIVAL_AGREEMENT_SAMPLES: f64 = 1.0;

/// Slack added to the room diagonal for the range plausibility gate (Q4).
///
/// Five meters covers microphone-position survey error plus an uncalibrated
/// source delay without admitting grossly wrong arrivals (reflections tens
/// of milliseconds late). Tighten only with a measured survey bound.
const RANGE_SLACK_M: f64 = 5.0;

/// Upper range bound when the plan declares no room dimensions (Q4).
///
/// Fifty meters admits halls while still refusing absurd arrivals from
/// mis-associated correlations. Sessions with declared dimensions use
/// the diagonal plus [`RANGE_SLACK_M`] instead.
const DEFAULT_MAX_RANGE_M: f64 = 50.0;

/// Maximum tolerated emitter-anchor disagreement in samples (Q6).
///
/// Compares the loopback-measured emitter propagation against the surveyed
/// emitter distance. Beyond one sample the survey and the wiring disagree
/// and the report trusts the electrical loopback, warning explicitly.
const EMITTER_ANCHOR_TOL_SAMPLES: f64 = 1.0;

/// Maximum loopback residual scatter across capture groups in samples.
///
/// The electrical path is identical for every group, so residuals beyond
/// one sample mean the loopback cable does not carry the stimulus and
/// loopback-referenced ranging is disabled session-wide.
const LOOPBACK_STABILITY_TOL_SAMPLES: f64 = 1.0;

/// Fallback speed of sound in meters per second.
///
/// Used only when neither a timing reference nor the plan declares one.
/// The ±3 m/s uncertainty covers roughly ±5 °C around 20 °C and visibly
/// widens every range uncertainty instead of silently biasing ranges.
const DEFAULT_SOUND_SPEED_M_S: f64 = 343.0;
/// See [`DEFAULT_SOUND_SPEED_M_S`].
const DEFAULT_SOUND_SPEED_UNCERTAINTY_M_S: f64 = 3.0;

/// Largest FFT the arrival correlator may allocate, in samples.
///
/// The envelope correlation pads to the next power of two over the sweep
/// plus the search window; beyond 2^23 the transient buffers exceed a few
/// hundred megabytes. Production sweeps (seconds) stay far below this;
/// longer stimuli refuse the cell instead of exhausting memory.
const MAX_CORRELATION_FFT: usize = 1 << 23;

/// Floor for the arrival standard deviation in samples.
///
/// Parabolic peak interpolation on a clean envelope resolves well below
/// one sample, but reporting exactly zero uncertainty is never honest;
/// one tenth of a sample (≈0.7 mm at 48 kHz) is the modeling floor.
const ARRIVAL_SIGMA_FLOOR_SAMPLES: f64 = 0.1;

/// What establishes the absolute time origin of a localization report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalizationMode {
    /// Ranges are microphone-minus-loopback on raw takes; the electrical
    /// loopback cancels clock offset, skew, and interface latency.
    LoopbackReferenced,
    /// Ranges come from raw arrivals mapped through the fitted clock whose
    /// offset excludes surveyed emitter propagation.
    EmitterReferenced,
    /// No absolute origin; only inter-microphone delays are published.
    RelativeOnly,
}

impl Display for LocalizationMode {
    /// Renders the mode as the CLI prints it.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoopbackReferenced => write!(formatter, "loopback-referenced"),
            Self::EmitterReferenced => write!(formatter, "emitter-referenced"),
            Self::RelativeOnly => write!(formatter, "relative-only"),
        }
    }
}

/// One source/microphone range cell with its evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RangeCell {
    /// Microphone identity from the session declaration.
    pub microphone_id: String,
    /// Absolute distance in meters; `None` means refused, see `reason`.
    #[serde(default)]
    pub distance_m: Option<f64>,
    /// One-sigma range uncertainty in meters; `None` means unknown.
    #[serde(default)]
    pub sigma_m: Option<f64>,
    /// Sweep-start arrival on the raw take in input-clock samples.
    #[serde(default)]
    pub arrival_samples: Option<f64>,
    /// Loopback sweep-start arrival for this cell's group in Mode A.
    #[serde(default)]
    pub reference_arrival_samples: Option<f64>,
    /// Delay relative to the group's first microphone in Mode C, samples.
    #[serde(default)]
    pub tdoa_samples_vs_first: Option<f64>,
    /// One-sigma uncertainty of the relative delay in samples.
    #[serde(default)]
    pub tdoa_sigma_samples: Option<f64>,
    /// Estimator band that produced this cell.
    #[serde(default)]
    pub band: Option<String>,
    /// Peak-to-sidelobe ratio of the arrival correlation in dB.
    #[serde(default)]
    pub confidence_db: Option<f64>,
    /// Stable machine token explaining a refused or relative-only cell.
    #[serde(default)]
    pub reason: Option<String>,
}

/// All microphone ranges for one source, in plan microphone order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceRanges {
    /// Source label from the session declaration.
    pub source_id: String,
    /// One cell per plan microphone, in plan order.
    pub cells: Vec<RangeCell>,
}

/// One estimated speaker position from multilateration over a range row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourcePosition {
    /// Source label from the session declaration.
    pub source_id: String,
    /// Estimated acoustic-centre position in meters, chirp band.
    #[serde(default)]
    pub position_m: Option<[f64; 3]>,
    /// Per-axis one-sigma uncertainty in meters.
    #[serde(default)]
    pub sigma_m: Option<[f64; 3]>,
    /// RMS range residual of the multilateration fit in meters.
    #[serde(default)]
    pub rms_residual_m: Option<f64>,
    /// Geometric dilution of precision of the microphone geometry.
    #[serde(default)]
    pub gdop: Option<f64>,
    /// True when coplanar microphones leave a mirror solution.
    pub mirror_ambiguous: bool,
    /// Distance to the surveyed source position, when declared.
    #[serde(default)]
    pub declared_error_m: Option<f64>,
    /// Stable machine token explaining a missing position.
    #[serde(default)]
    pub reason: Option<String>,
}

/// Session range matrix with its time-origin evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalizationReport {
    /// Session-wide time-origin mode; every absolute cell shares it.
    pub mode: LocalizationMode,
    /// Speed of sound used for delay-to-distance conversion.
    pub sound_speed_m_s: f64,
    /// Upper bound on sound-speed error in meters per second.
    pub sound_speed_uncertainty_m_s: f64,
    /// Where the sound speed came from: timing reference, plan, default.
    pub sound_speed_source: String,
    /// Median fitted skew across takes in ppm; the outlier reference.
    #[serde(default)]
    pub pooled_skew_ppm: Option<f64>,
    /// Microphone common-bias correction applied to every range in meters.
    pub mic_bias_correction_m: f64,
    /// One row per plan source, in plan order.
    pub ranges: Vec<SourceRanges>,
    /// One entry per plan source, aligned with `ranges`.
    #[serde(default)]
    pub sources: Vec<SourcePosition>,
    /// Session-level evidence notes and degradation warnings.
    #[serde(default)]
    pub issues: Vec<String>,
}

/// Sweep arrival measured on one raw recording.
#[derive(Debug, Clone, Copy)]
struct SweepArrival {
    /// Direct (first significant) sweep-start index in input samples.
    direct_samples: f64,
    /// Peak-to-sidelobe ratio of the envelope correlation in dB.
    confidence_db: f64,
    /// One-sigma arrival uncertainty in samples (may be infinite).
    sigma_samples: f64,
    /// Sub-band the arrival was estimated in, in Hz.
    band_hz: (f64, f64),
}

/// Resolve the session speed of sound and its provenance.
///
/// A timing reference wins when declared; otherwise the plan value;
/// otherwise the documented default with an honest uncertainty.
fn resolve_sound_speed(plan: &CaptureSessionPlan) -> (f64, f64, &'static str) {
    if let Some(reference) = &plan.timing_reference {
        return (
            reference.sound_speed_m_s,
            reference.sound_speed_uncertainty_m_s,
            "timing_reference",
        );
    }
    if let Some(speed) = &plan.sound_speed {
        return (speed.m_s, speed.uncertainty_m_s, "plan");
    }
    (
        DEFAULT_SOUND_SPEED_M_S,
        DEFAULT_SOUND_SPEED_UNCERTAINTY_M_S,
        "default",
    )
}

/// Median of fitted skews, the single-clock pool reference.
///
/// All takes share one ADC clock, so the median skew is the session
/// rate and takes disagreeing with it are clock outliers (Q1). The
/// upper middle keeps the result an actual fitted value.
fn pooled_skew_ppm(skews: &mut [f64]) -> Option<f64> {
    if skews.is_empty() || skews.iter().any(|skew| !skew.is_finite()) {
        return None;
    }
    skews.sort_by(f64::total_cmp);
    Some(skews[skews.len() / 2])
}

/// One-sigma arrival uncertainty from correlation confidence.
///
/// Delay-estimation Cramér–Rao form with peak-to-sidelobe ratio as the
/// SNR proxy and rectangular-band RMS bandwidth, mirroring the clock
/// stage model: `σ = Fs / (2π·β·√(2·SNR))`. Infinite confidence (a
/// noiseless correlation) caps at 100 dB, where the interpolation floor
/// dominates anyway; anything else non-finite returns infinity and
/// callers report unknown, never zero.
fn arrival_sigma_samples(confidence_db: f64, band_lo_hz: f64, band_hi_hz: f64, rate: f64) -> f64 {
    let confidence_db = if confidence_db == f64::INFINITY {
        100.0
    } else {
        confidence_db
    };
    let bandwidth = band_hi_hz - band_lo_hz;
    if !confidence_db.is_finite()
        || !bandwidth.is_finite()
        || bandwidth <= 0.0
        || !rate.is_finite()
        || rate <= 0.0
    {
        return f64::INFINITY;
    }
    let snr = 10.0_f64.powf(confidence_db / 10.0);
    if !snr.is_finite() || snr <= 0.0 {
        return f64::INFINITY;
    }
    let beta = bandwidth / 12.0_f64.sqrt();
    let sigma = rate / (2.0 * std::f64::consts::PI * beta * (2.0 * snr).sqrt());
    if sigma.is_finite() {
        sigma.max(ARRIVAL_SIGMA_FLOOR_SAMPLES)
    } else {
        f64::INFINITY
    }
}

/// Predict the sweep-start index on a raw recording from its clock fit.
///
/// The start-chirp lag anchors the prediction; the sweep offset follows
/// on the fitted skew ratio. Callers window the correlation around this
/// prediction instead of scanning the whole recording.
#[expect(
    clippy::cast_precision_loss,
    reason = "Stimulus offsets convert to f64 sample coordinates; bounded layouts stay far below precision limits."
)]
fn predict_sweep_start(
    fit: &CaptureClockFit,
    start_chirp_offset: usize,
    sweep_offset: usize,
) -> f64 {
    let ratio = 1.0 + fit.skew.skew_ppm / 1e6;
    let first_raw = fit.first.offset_samples + start_chirp_offset as f64;
    first_raw + (sweep_offset.saturating_sub(start_chirp_offset)) as f64 * ratio
}

/// Design band for arrival estimation in Hz.
///
/// Arrivals correlate only this sub-band of the sweep: a full-range log
/// sweep's decades-wide asymmetric spectrum biases the envelope peak by
/// fractions of a sample, while a symmetric kilohertz-wide band keeps
/// the bias an order of magnitude below the agreement gate. Sweeps that
/// do not cover enough of this band refuse with `band_unsupported`
/// (subwoofers get their own low-band estimator in Phase 3).
const ARRIVAL_BAND_LO_HZ: f64 = 2000.0;
/// See [`ARRIVAL_BAND_LO_HZ`].
const ARRIVAL_BAND_HI_HZ: f64 = 8000.0;
/// Minimum usable arrival bandwidth in Hz (Q-gate for `band_unsupported`).
const MIN_ARRIVAL_BANDWIDTH_HZ: f64 = 2000.0;
/// Arrival bandpass length in taps (linear phase, odd, symmetric).
const ARRIVAL_FIR_TAPS: usize = 127;

/// Usable arrival band for one sweep, intersected with the design band.
///
/// Returns `None` when the sweep leaves less than
/// [`MIN_ARRIVAL_BANDWIDTH_HZ`] inside the design band: narrow or
/// out-of-band sweeps cannot produce an unbiased arrival here.
fn arrival_band(sweep_lo_hz: f64, sweep_hi_hz: f64) -> Option<(f64, f64)> {
    if !sweep_lo_hz.is_finite() || !sweep_hi_hz.is_finite() || sweep_hi_hz <= sweep_lo_hz {
        return None;
    }
    let lo = sweep_lo_hz.max(ARRIVAL_BAND_LO_HZ);
    let hi = sweep_hi_hz.min(ARRIVAL_BAND_HI_HZ);
    if hi - lo >= MIN_ARRIVAL_BANDWIDTH_HZ {
        Some((lo, hi))
    } else {
        None
    }
}

/// Windowed-sinc linear-phase bandpass for arrival estimation.
///
/// Both the reference sweep and the recorded window pass through this
/// identical filter, so its constant group delay cancels in the lag and
/// the correlation sees a symmetric sub-band with no phase distortion.
#[expect(
    clippy::cast_precision_loss,
    reason = "The fixed 127-tap count converts to f64 exactly; far below precision limits."
)]
fn arrival_bandpass(rate: f64, band_lo_hz: f64, band_hi_hz: f64) -> Vec<f64> {
    // Integer center: with odd length the sinc and the Hann window share
    // tap (N-1)/2, which is what makes the phase exactly linear.
    let middle = (ARRIVAL_FIR_TAPS - 1) as f64 / 2.0;
    (0..ARRIVAL_FIR_TAPS)
        .map(|tap| {
            let offset = tap as f64 - middle;
            let sinc = if offset.abs() < 1e-9 {
                2.0 * (band_hi_hz - band_lo_hz) / rate
            } else {
                let phase = std::f64::consts::PI * offset / rate;
                ((2.0 * band_hi_hz * phase).sin() - (2.0 * band_lo_hz * phase).sin())
                    / (std::f64::consts::PI * offset)
            };
            let window = 0.5
                - 0.5
                    * (2.0 * std::f64::consts::PI * tap as f64 / (ARRIVAL_FIR_TAPS - 1) as f64)
                        .cos();
            sinc * window
        })
        .collect()
}

/// Filter one signal with the arrival bandpass, same length out.
///
/// Direct-form convolution centered on each sample; edge transients
/// stay within half the filter length of the ends, far from arrivals.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Convolution indices stay within the bounded signal; conversions round-trip exactly."
)]
fn apply_arrival_bandpass(signal: &[f32], taps: &[f64]) -> Vec<f32> {
    let pad = taps.len() / 2;
    (0..signal.len())
        .map(|n| {
            let mut acc = 0.0;
            for (k, tap) in taps.iter().enumerate() {
                let index = n as isize + k as isize - pad as isize;
                if index >= 0 && (index as usize) < signal.len() {
                    acc += f64::from(signal[index as usize]) * tap;
                }
            }
            acc as f32
        })
        .collect()
}

/// Estimate one sweep arrival on a raw recording.
///
/// Band-limits the known sweep and a window around the fitted prediction
/// to a symmetric sub-band, then correlates: the direct (first
/// significant) and strongest (global maximum) envelope peaks must agree
/// within one sample (Q3). Refusals carry stable reason tokens: a missing
/// window, an unsupported band, a weak correlation
/// (`arrival_low_confidence`, Q2), or disagreeing direct/strongest
/// estimators (`arrival_ambiguous`, Q3).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "Nonnegative predicted windows round to whole frames and clamp to the recording."
)]
fn estimate_sweep_arrival(
    sweep: &[f32],
    recorded: &[f32],
    predicted_start: f64,
    rate: u32,
    sweep_lo_hz: f64,
    sweep_hi_hz: f64,
) -> Result<SweepArrival, &'static str> {
    if sweep.is_empty() || recorded.is_empty() || !predicted_start.is_finite() {
        return Err("sweep_window_missing");
    }
    let Some((band_lo, band_hi)) = arrival_band(sweep_lo_hz, sweep_hi_hz) else {
        return Err("band_unsupported");
    };
    let margin = (f64::from(rate) / 4.0)
        .min(sweep.len() as f64 / 2.0)
        .max(64.0);
    let start = (predicted_start - margin).max(0.0) as usize;
    let stop = (predicted_start + sweep.len() as f64 + margin)
        .min(recorded.len() as f64)
        .max(0.0) as usize;
    let window = recorded.get(start..stop).unwrap_or_default();
    if window.len() < sweep.len().min(64) || window.len() < 64 {
        return Err("sweep_window_missing");
    }
    let fft = (sweep.len() + window.len()).next_power_of_two();
    if fft > MAX_CORRELATION_FFT {
        return Err("sweep_too_long");
    }
    let taps = arrival_bandpass(f64::from(rate), band_lo, band_hi);
    let reference = apply_arrival_bandpass(sweep, &taps);
    let windowed = apply_arrival_bandpass(window, &taps);
    let correlation = cross_correlate_envelope(&reference, &windowed, rate)
        .map_err(|_| "arrival_low_confidence")?;
    let confidence_db = f64::from(correlation.peak_to_sidelobe_db);
    if confidence_db.is_nan() || confidence_db < MIN_TDOA_CONFIDENCE_DB {
        return Err("arrival_low_confidence");
    }
    let direct = crate::signal_recorder::pick_direct_arrival_from_envelope(
        &correlation.envelope,
        rate,
        windowed.len(),
    )
    .ok_or("arrival_low_confidence")?;
    let strongest = start as f64 + correlation.peak_sample_refined;
    let direct_at = start as f64 + direct.peak_sample_refined;
    if (direct_at - strongest).abs() > ARRIVAL_AGREEMENT_SAMPLES {
        return Err("arrival_ambiguous");
    }
    Ok(SweepArrival {
        direct_samples: direct_at,
        confidence_db,
        sigma_samples: arrival_sigma_samples(confidence_db, band_lo, band_hi, f64::from(rate)),
        band_hz: (band_lo, band_hi),
    })
}

/// Upper plausibility bound for ranges in meters (Q4).
///
/// The room diagonal plus slack when dimensions are declared, otherwise
/// the generous default. Never smaller than the slack itself.
fn max_range_m(plan: &CaptureSessionPlan) -> f64 {
    plan.room_dimensions
        .map(|room| {
            (room.length.powi(2) + room.width.powi(2) + room.height.powi(2)).sqrt() + RANGE_SLACK_M
        })
        .filter(|bound| bound.is_finite() && *bound > 0.0)
        .unwrap_or(DEFAULT_MAX_RANGE_M)
}

/// Convert acoustic propagation samples into meters.
///
/// Subtracts the declared source electronic delay when present. The
/// matched-microphone common bias applies from Phase 3; until then the
/// report states the zero correction explicitly.
fn propagation_to_meters(
    prop_samples: f64,
    rate: f64,
    sound_speed: f64,
    source: &CaptureSource,
) -> Option<f64> {
    if !prop_samples.is_finite() || !rate.is_finite() || rate <= 0.0 {
        return None;
    }
    let mut prop = prop_samples;
    if let Some(delay_ms) = source.source_delay_calibration_ms {
        prop -= delay_ms / 1000.0 * rate;
    }
    let distance = prop / rate * sound_speed;
    distance.is_finite().then_some(distance)
}

/// Combine timing uncertainty with sound-speed uncertainty in meters.
///
/// Worst-cased sum, mirroring the clock stage: arrival/clock timing
/// converted at the sound speed plus the fractional sound-speed error
/// times the distance. Returns `None` when timing is unknown.
fn range_sigma_m(
    sigma_time_samples: f64,
    distance: f64,
    rate: f64,
    sound_speed: f64,
    sound_speed_uncertainty: f64,
) -> Option<f64> {
    if !sigma_time_samples.is_finite() || sigma_time_samples < 0.0 {
        return None;
    }
    let sigma =
        sound_speed * sigma_time_samples / rate + distance * sound_speed_uncertainty / sound_speed;
    sigma.is_finite().then_some(sigma)
}

/// Estimate one take's sweep arrival with clock and quality gates.
///
/// Applies Q1 (usable fit, pooled-skew outlier) and Q9 (broadband SNR)
/// before correlating, then the estimator's own Q2/Q3 gates. Reads the
/// take's raw device-clock recording; corrected audio is never needed
/// because the fit maps raw arrivals onto the stimulus clock.
fn take_sweep_arrival(
    report: &ClockProcessedManifest,
    output: &Path,
    take_index: usize,
    fits: &[Option<CaptureClockFit>],
    pooled: Option<f64>,
    sweep: &[f32],
) -> Result<SweepArrival, &'static str> {
    let take = report.takes.get(take_index).ok_or("take_unreadable")?;
    let fit = fits
        .get(take_index)
        .copied()
        .flatten()
        .ok_or("clock_uncorrected")?;
    if take.clock.basis == CaptureClockBasis::Uncorrected {
        return Err("clock_uncorrected");
    }
    match pooled {
        Some(pooled) if (fit.skew.skew_ppm - pooled).abs() <= CLOCK_SKEW_SPREAD_PPM => {}
        _ => return Err("clock_outlier"),
    }
    if let Some(analysis) = take.analysis.as_ref()
        && let Some(snr) = analysis.broadband_snr_db
        && snr < COHERENT_CAPTURE_SNR_THRESHOLD_DB
    {
        return Err("band_snr_low");
    }
    let rate = report.plan.sample_rate_hz;
    let path =
        super::clock::io::artifact(output, &take.raw.wav_file).map_err(|_| "take_unreadable")?;
    let recorded = super::clock::io::read_wave(&path, rate, take.raw.samples)
        .map_err(|_| "take_unreadable")?;
    let layout = &report.stimulus;
    let predicted = predict_sweep_start(&fit, layout.start_chirp_offset, layout.sweep_offset);
    estimate_sweep_arrival(
        sweep,
        &recorded,
        predicted,
        rate,
        report.plan.sweep.start_hz,
        report.plan.sweep.end_hz,
    )
}

/// Scatter of electrical loopback residuals across groups.
///
/// Each residual subtracts the group's fitted prediction from the
/// loopback's measured arrival, leaving the electrical path plus the
/// common microphone's emitter travel: constant across groups when the
/// cable carries the stimulus. Scatter beyond one sample invalidates
/// loopback-referenced ranging session-wide.
fn loopback_residual_scatter(residuals: &[f64]) -> Option<f64> {
    if residuals.is_empty() || residuals.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let scatter = residuals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        - residuals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    scatter.is_finite().then_some(scatter)
}

/// One-line evidence note describing the decided session mode.
fn mode_evidence(
    mode: LocalizationMode,
    groups: usize,
    with_loopback: usize,
    sound: &str,
) -> String {
    match mode {
        LocalizationMode::LoopbackReferenced => format!(
            "localization mode loopback-referenced: {with_loopback}/{groups} selected groups with loopback takes, sound speed from {sound}"
        ),
        LocalizationMode::EmitterReferenced => format!(
            "localization mode emitter-referenced: fixed-acoustic-reference takes, sound speed from {sound}"
        ),
        LocalizationMode::RelativeOnly => format!(
            "localization mode relative-only: no session-wide absolute origin, sound speed from {sound}"
        ),
    }
}

/// Takes selected for one source: a single capture group at most.
///
/// Differential measurements (microphone minus loopback or microphone)
/// require one shared capture call, so every mode works within the
/// fullest group (lowest repeat on ties). Repeat medians arrive in
/// Phase 3; until then other repeats are ignored, never averaged.
#[derive(Debug, Clone)]
struct SelectedGroup {
    source_index: usize,
    repeat_index: u32,
    /// `report.takes` indices in take order.
    take_indices: Vec<usize>,
    /// `report.loopback_files` index for this source and repeat, if any.
    loopback_index: Option<usize>,
}

/// Every (source, repeat) capture group present in the report.
///
/// Ordered by source plan order, then repeat index. Groups share one
/// capture call internally: one startup offset, one skew, one loopback.
fn capture_groups(report: &ClockProcessedManifest) -> Vec<SelectedGroup> {
    let mut grouped: BTreeMap<(usize, u32), Vec<usize>> = BTreeMap::new();
    for (take_index, take) in report.takes.iter().enumerate() {
        let Some(source_index) = report
            .plan
            .sources
            .iter()
            .position(|source| source.id == take.raw.source_id)
        else {
            continue;
        };
        grouped
            .entry((source_index, take.raw.repeat_index))
            .or_default()
            .push(take_index);
    }
    grouped
        .into_iter()
        .map(|((source_index, repeat_index), take_indices)| {
            let source = &report.plan.sources[source_index];
            let loopback_index = report.loopback_files.iter().position(|loopback| {
                loopback.source_id == source.id && loopback.repeat_index == repeat_index
            });
            SelectedGroup {
                source_index,
                repeat_index,
                take_indices,
                loopback_index,
            }
        })
        .collect()
}

/// Select one capture group per plan source, in plan order.
///
/// The fullest group wins; the lowest repeat breaks ties. Sources
/// without takes get an empty group so their row still reports
/// `take_missing` per microphone.
fn select_groups(report: &ClockProcessedManifest) -> Vec<SelectedGroup> {
    let all = capture_groups(report);
    report
        .plan
        .sources
        .iter()
        .enumerate()
        .map(|(source_index, source)| {
            all.iter()
                .filter(|group| group.source_index == source_index)
                .max_by(|a, b| {
                    a.take_indices
                        .len()
                        .cmp(&b.take_indices.len())
                        .then(b.repeat_index.cmp(&a.repeat_index))
                })
                .cloned()
                .unwrap_or(SelectedGroup {
                    source_index,
                    repeat_index: 0,
                    take_indices: Vec::new(),
                    loopback_index: report.loopback_files.iter().position(|loopback| {
                        loopback.source_id == source.id && loopback.repeat_index == 0
                    }),
                })
        })
        .collect()
}

/// Usable electrical reference for one capture group.
struct GroupLoopback {
    arrival: SweepArrival,
    device_id: String,
}

/// Estimate one group's loopback arrival from the raw journal.
///
/// The loopback shares its capture call with the group's microphones,
/// so the first fitted microphone take predicts the sweep window; the
/// loopback needs no independent clock fit. Failures explain themselves
/// for the session issues; cells simply report `loopback_unavailable`.
fn group_loopback_arrival(
    report: &ClockProcessedManifest,
    raw_root: &Path,
    group: &SelectedGroup,
    fits: &[Option<CaptureClockFit>],
    sweep: &[f32],
) -> Result<GroupLoopback, String> {
    let Some(loopback_index) = group.loopback_index else {
        return Err("no loopback take for this source and repeat".into());
    };
    let loopback = &report.loopback_files[loopback_index];
    let Some(fit) = group
        .take_indices
        .iter()
        .find_map(|take| fits.get(*take).copied().flatten())
    else {
        return Err("no fitted microphone take predicts the loopback window".into());
    };
    let rate = report.plan.sample_rate_hz;
    let path = super::clock::io::artifact(raw_root, &loopback.wav_file)
        .map_err(|error| format!("loopback artifact rejected: {error}"))?;
    let recorded = super::clock::io::read_wave(&path, rate, loopback.samples)
        .map_err(|error| format!("loopback audio unreadable: {error}"))?;
    let layout = &report.stimulus;
    let predicted = predict_sweep_start(&fit, layout.start_chirp_offset, layout.sweep_offset);
    let arrival = estimate_sweep_arrival(
        sweep,
        &recorded,
        predicted,
        rate,
        report.plan.sweep.start_hz,
        report.plan.sweep.end_hz,
    )
    .map_err(|token| format!("loopback arrival refused: {token}"))?;
    Ok(GroupLoopback {
        arrival,
        device_id: loopback.device_id.clone(),
    })
}

/// Check the emitter survey against loopback-measured propagation (Q6).
///
/// With both an electrical loopback and a surveyed emitter, each
/// microphone's emitter travel is measured independently of the survey.
/// The worst disagreement beyond one sample warns explicitly; the
/// session keeps trusting the loopback, never the survey.
#[expect(
    clippy::cast_precision_loss,
    reason = "Surveyed distances convert to f64 sample counts; room-scale geometry is far below precision limits."
)]
fn emitter_anchor_check(
    report: &ClockProcessedManifest,
    group: &SelectedGroup,
    loopback: &GroupLoopback,
    fits: &[Option<CaptureClockFit>],
    arrivals: &BTreeMap<usize, Result<SweepArrival, &'static str>>,
) -> Option<String> {
    let emitter = report.plan.timing_reference.as_ref()?;
    let rate = f64::from(report.plan.sample_rate_hz);
    let layout = &report.stimulus;
    let mut worst = 0.0_f64;
    let mut worst_mic = "";
    for take_index in &group.take_indices {
        let take = &report.takes[*take_index];
        if take.raw.device_id != loopback.device_id {
            continue;
        }
        let (Some(fit), Some(Ok(_))) = (
            fits.get(*take_index).copied().flatten(),
            arrivals.get(take_index),
        ) else {
            continue;
        };
        let Some(mic) = report
            .plan
            .microphones
            .iter()
            .find(|mic| mic.id == take.raw.microphone_id)
        else {
            continue;
        };
        let ratio = 1.0 + fit.skew.skew_ppm / 1e6;
        let electrical = loopback.arrival.direct_samples - ratio * layout.sweep_offset as f64;
        let measured = fit.first.offset_samples
            - (ratio - 1.0) * layout.start_chirp_offset as f64
            - electrical;
        let surveyed = mic
            .position_m
            .iter()
            .zip(emitter.position_m)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            / emitter.sound_speed_m_s
            * rate;
        let disagreement = (measured - surveyed).abs();
        if disagreement > worst {
            worst = disagreement;
            worst_mic = mic.id.as_str();
        }
    }
    if worst > EMITTER_ANCHOR_TOL_SAMPLES {
        let source = &report.plan.sources[group.source_index].id;
        Some(format!(
            "emitter_mismatch on {source}/{worst_mic}: loopback-measured emitter travel disagrees with the survey by {worst:.1} samples; trusting the loopback"
        ))
    } else {
        None
    }
}

/// Empty range cell reporting one refusal token.
fn refused_cell(microphone_id: &str, reason: &'static str) -> RangeCell {
    RangeCell {
        microphone_id: microphone_id.into(),
        distance_m: None,
        sigma_m: None,
        arrival_samples: None,
        reference_arrival_samples: None,
        tdoa_samples_vs_first: None,
        tdoa_sigma_samples: None,
        band: None,
        confidence_db: None,
        reason: Some(reason.into()),
    }
}

/// Build one source row in loopback-referenced mode (A).
///
/// Propagation is the microphone arrival minus the loopback arrival on
/// raw takes of one capture call, scaled by the pooled session skew
/// ratio: clock offset and interface latency cancel without resampling.
#[expect(
    clippy::too_many_arguments,
    reason = "Mode conversion threads the decided session evidence explicitly; bundling would hide the per-cell inputs."
)]
fn loopback_cells(
    report: &ClockProcessedManifest,
    group: &SelectedGroup,
    loopback: &GroupLoopback,
    arrivals: &BTreeMap<usize, Result<SweepArrival, &'static str>>,
    sound_speed: f64,
    sound_speed_uncertainty: f64,
    max_range: f64,
    skew_ratio: f64,
) -> SourceRanges {
    let source = &report.plan.sources[group.source_index];
    let rate = f64::from(report.plan.sample_rate_hz);
    let mut cells = Vec::new();
    for mic in &report.plan.microphones {
        let Some(take_index) = group
            .take_indices
            .iter()
            .find(|index| report.takes[**index].raw.microphone_id == mic.id)
        else {
            cells.push(refused_cell(&mic.id, "take_missing"));
            continue;
        };
        let take = &report.takes[*take_index];
        if take.raw.device_id != loopback.device_id {
            cells.push(refused_cell(&mic.id, "loopback_unavailable"));
            continue;
        }
        let arrival = match arrivals.get(take_index) {
            Some(Ok(arrival)) => *arrival,
            Some(Err(token)) => {
                cells.push(refused_cell(&mic.id, token));
                continue;
            }
            None => {
                cells.push(refused_cell(&mic.id, "take_missing"));
                continue;
            }
        };
        let prop = (arrival.direct_samples - loopback.arrival.direct_samples) / skew_ratio;
        let Some(distance) = propagation_to_meters(prop, rate, sound_speed, source) else {
            cells.push(refused_cell(&mic.id, "range_implausible"));
            continue;
        };
        if distance < 0.0 || distance > max_range {
            cells.push(refused_cell(&mic.id, "range_implausible"));
            continue;
        }
        let sigma_time = (arrival.sigma_samples.powi(2) + loopback.arrival.sigma_samples.powi(2))
            .sqrt()
            / skew_ratio;
        cells.push(RangeCell {
            microphone_id: mic.id.clone(),
            distance_m: Some(distance),
            sigma_m: range_sigma_m(
                sigma_time,
                distance,
                rate,
                sound_speed,
                sound_speed_uncertainty,
            ),
            arrival_samples: Some(arrival.direct_samples),
            reference_arrival_samples: Some(loopback.arrival.direct_samples),
            tdoa_samples_vs_first: None,
            tdoa_sigma_samples: None,
            band: Some(band_label(arrival.band_hz)),
            confidence_db: finite_confidence(arrival.confidence_db),
            reason: None,
        });
    }
    SourceRanges {
        source_id: source.id.clone(),
        cells,
    }
}

/// Build one source row in emitter-referenced mode (B).
///
/// Raw arrivals map through the fitted clock whose offset excludes the
/// surveyed emitter propagation, so the residual is the source travel.
/// Takes without the fixed-acoustic-reference basis never reach here:
/// the session mode decision already excluded them.
#[expect(
    clippy::cast_precision_loss,
    reason = "Sample counts convert to f64 meters; bounded recordings stay far below precision limits."
)]
fn emitter_cells(
    report: &ClockProcessedManifest,
    group: &SelectedGroup,
    fits: &[Option<CaptureClockFit>],
    arrivals: &BTreeMap<usize, Result<SweepArrival, &'static str>>,
    sound_speed: f64,
    sound_speed_uncertainty: f64,
    max_range: f64,
) -> SourceRanges {
    let source = &report.plan.sources[group.source_index];
    let rate = f64::from(report.plan.sample_rate_hz);
    let sweep_offset = report.stimulus.sweep_offset as f64;
    let mut cells = Vec::new();
    for mic in &report.plan.microphones {
        let Some(take_index) = group
            .take_indices
            .iter()
            .find(|index| report.takes[**index].raw.microphone_id == mic.id)
        else {
            cells.push(refused_cell(&mic.id, "take_missing"));
            continue;
        };
        let take = &report.takes[*take_index];
        let arrival = match arrivals.get(take_index) {
            Some(Ok(arrival)) => *arrival,
            Some(Err(token)) => {
                cells.push(refused_cell(&mic.id, token));
                continue;
            }
            None => {
                cells.push(refused_cell(&mic.id, "take_missing"));
                continue;
            }
        };
        let Some(fit) = fits.get(*take_index).copied().flatten() else {
            cells.push(refused_cell(&mic.id, "clock_uncorrected"));
            continue;
        };
        let ratio = 1.0 + fit.skew.skew_ppm / 1e6;
        let prop = (arrival.direct_samples - fit.skew.offset_samples) / ratio - sweep_offset;
        let Some(distance) = propagation_to_meters(prop, rate, sound_speed, source) else {
            cells.push(refused_cell(&mic.id, "range_implausible"));
            continue;
        };
        if distance < 0.0 || distance > max_range {
            cells.push(refused_cell(&mic.id, "range_implausible"));
            continue;
        }
        let clock_samples = take
            .clock
            .residual_uncertainty_us
            .or(take.clock.marker_uncertainty_us)
            .map(|bound| bound / 1e6 * rate)
            .filter(|bound| bound.is_finite() && *bound >= 0.0);
        let sigma_time = clock_samples.map_or(f64::INFINITY, |clock| {
            (arrival.sigma_samples.powi(2) + clock.powi(2)).sqrt()
        });
        cells.push(RangeCell {
            microphone_id: mic.id.clone(),
            distance_m: Some(distance),
            sigma_m: range_sigma_m(
                sigma_time,
                distance,
                rate,
                sound_speed,
                sound_speed_uncertainty,
            ),
            arrival_samples: Some(arrival.direct_samples),
            reference_arrival_samples: None,
            tdoa_samples_vs_first: None,
            tdoa_sigma_samples: None,
            band: Some(band_label(arrival.band_hz)),
            confidence_db: finite_confidence(arrival.confidence_db),
            reason: None,
        });
    }
    SourceRanges {
        source_id: source.id.clone(),
        cells,
    }
}

/// Build one source row in relative-only mode (C).
///
/// Delays are microphone-minus-first-microphone on raw takes of one
/// capture call, scaled to stimulus-clock samples by the pooled skew
/// ratio: shared clock terms cancel, but no absolute origin exists, so
/// every distance stays refused with its reason.
fn relative_cells(
    report: &ClockProcessedManifest,
    group: &SelectedGroup,
    arrivals: &BTreeMap<usize, Result<SweepArrival, &'static str>>,
    skew_ratio: f64,
) -> SourceRanges {
    let source = &report.plan.sources[group.source_index];
    let reference = report.plan.microphones.iter().find_map(|mic| {
        group
            .take_indices
            .iter()
            .find(|index| report.takes[**index].raw.microphone_id == mic.id)
            .and_then(|index| arrivals.get(index))
            .and_then(|arrival| arrival.ok())
            .map(|arrival| (mic.id.clone(), arrival))
    });
    let mut cells = Vec::new();
    for mic in &report.plan.microphones {
        let Some(take_index) = group
            .take_indices
            .iter()
            .find(|index| report.takes[**index].raw.microphone_id == mic.id)
        else {
            cells.push(refused_cell(&mic.id, "take_missing"));
            continue;
        };
        let arrival = match arrivals.get(take_index) {
            Some(Ok(arrival)) => *arrival,
            Some(Err(token)) => {
                cells.push(refused_cell(&mic.id, token));
                continue;
            }
            None => {
                cells.push(refused_cell(&mic.id, "take_missing"));
                continue;
            }
        };
        let Some((_, first)) = reference else {
            cells.push(refused_cell(&mic.id, "arrival_low_confidence"));
            continue;
        };
        let delay = (arrival.direct_samples - first.direct_samples) / skew_ratio;
        let sigma = if arrival.sigma_samples.is_finite() && first.sigma_samples.is_finite() {
            Some((arrival.sigma_samples.powi(2) + first.sigma_samples.powi(2)).sqrt() / skew_ratio)
        } else {
            None
        };
        cells.push(RangeCell {
            microphone_id: mic.id.clone(),
            distance_m: None,
            sigma_m: None,
            arrival_samples: Some(arrival.direct_samples),
            reference_arrival_samples: None,
            tdoa_samples_vs_first: Some(delay),
            tdoa_sigma_samples: sigma,
            band: Some(band_label(arrival.band_hz)),
            confidence_db: finite_confidence(arrival.confidence_db),
            reason: Some("no_absolute_time_origin".into()),
        });
    }
    SourceRanges {
        source_id: source.id.clone(),
        cells,
    }
}

/// Render the arrival sub-band as the stored band label.
fn band_label(band_hz: (f64, f64)) -> String {
    let (lo, hi) = band_hz;
    format!("sweep_{lo:.0}_{hi:.0}")
}

/// Cap correlation confidence for persistence.
///
/// Peak-to-sidelobe ratios above 100 dB sit far past the interpolation
/// floor that dominates the uncertainty model; capping keeps stored
/// JSON finite while preserving the ordering of real measurements.
fn finite_confidence(confidence_db: f64) -> Option<f64> {
    if confidence_db.is_nan() {
        return None;
    }
    Some(confidence_db.min(100.0))
}

/// Verify loopback consistency across capture groups.
///
/// Each residual subtracts one group's fitted prediction from its
/// loopback arrival. Without a surveyed emitter the chirp path follows
/// the playing source, so residuals are comparable only within one
/// source across its repeats (a glitch check); with a fixed emitter
/// the path is source-independent and all residuals pool into one
/// cable-identity check. Either scatter beyond one sample disables
/// loopback ranging. Returns true when stable or unverifiable, with
/// the reason recorded in the session issues either way.
#[expect(
    clippy::too_many_lines,
    reason = "The stability check keeps per-source and emitter-gated evidence in one ordered pass."
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Stimulus offsets convert to f64 sample coordinates; bounded layouts stay far below precision limits."
)]
fn loopback_stable(
    report: &ClockProcessedManifest,
    all_groups: &[SelectedGroup],
    loopbacks: &BTreeMap<(usize, u32), Result<GroupLoopback, String>>,
    fits: &[Option<CaptureClockFit>],
    issues: &mut Vec<String>,
) -> bool {
    let sweep_offset = report.stimulus.sweep_offset as f64;
    let usable: Vec<&SelectedGroup> = all_groups
        .iter()
        .filter(|group| {
            loopbacks
                .get(&(group.source_index, group.repeat_index))
                .is_some_and(Result::is_ok)
        })
        .collect();
    if usable.len() < 2 {
        issues.push("loopback stability unverified: fewer than two usable loopback groups".into());
        return true;
    }
    let residual = |group: &SelectedGroup, microphone_id: &str| -> Option<f64> {
        let take_index = group
            .take_indices
            .iter()
            .find(|take| report.takes[**take].raw.microphone_id == microphone_id)?;
        let take = &report.takes[*take_index];
        let Ok(loopback) = loopbacks.get(&(group.source_index, group.repeat_index))? else {
            return None;
        };
        if take.raw.device_id != loopback.device_id {
            return None;
        }
        let fit = fits.get(*take_index).copied().flatten()?;
        let ratio = 1.0 + fit.skew.skew_ppm / 1e6;
        Some(loopback.arrival.direct_samples - fit.skew.offset_samples - ratio * sweep_offset)
    };
    let common_microphone = |groups: &[&SelectedGroup]| -> Option<&str> {
        report.plan.microphones.iter().find_map(|mic| {
            groups
                .iter()
                .all(|group| residual(group, &mic.id).is_some())
                .then_some(mic.id.as_str())
        })
    };
    for source_index in 0..report.plan.sources.len() {
        let source_groups: Vec<&SelectedGroup> = usable
            .iter()
            .filter(|group| group.source_index == source_index)
            .copied()
            .collect();
        if source_groups.len() < 2 {
            continue;
        }
        let Some(microphone) = common_microphone(&source_groups) else {
            issues.push(format!(
                "loopback stability unverified for source {}: no shared fitted microphone",
                report.plan.sources[source_index].id
            ));
            continue;
        };
        let residuals: Vec<f64> = source_groups
            .iter()
            .filter_map(|group| residual(group, microphone))
            .collect();
        match loopback_residual_scatter(&residuals) {
            Some(scatter) if scatter > LOOPBACK_STABILITY_TOL_SAMPLES => {
                issues.push(format!(
                    "loopback_unstable: electrical residuals of source {} scatter {scatter:.1} samples across repeats; loopback ranging disabled",
                    report.plan.sources[source_index].id
                ));
                return false;
            }
            Some(scatter) => issues.push(format!(
                "loopback residuals of source {} scatter {scatter:.2} samples across {} repeats",
                report.plan.sources[source_index].id,
                residuals.len()
            )),
            None => {}
        }
    }
    if report.plan.timing_reference.is_none() {
        issues.push(
            "loopback cable identity unverified without a surveyed emitter; arrival confidence and downstream geometry are the remaining guards"
                .into(),
        );
        return true;
    }
    let Some(microphone) = common_microphone(&usable) else {
        issues.push(
            "loopback cable identity unverified: no microphone is fitted in every loopback group"
                .into(),
        );
        return true;
    };
    let residuals: Vec<f64> = usable
        .iter()
        .filter_map(|group| residual(group, microphone))
        .collect();
    match loopback_residual_scatter(&residuals) {
        Some(scatter) if scatter > LOOPBACK_STABILITY_TOL_SAMPLES => {
            issues.push(format!(
                "loopback_unstable: electrical residuals scatter {scatter:.1} samples across sources; loopback ranging disabled"
            ));
            false
        }
        Some(scatter) => {
            issues.push(format!(
                "loopback residuals scatter {scatter:.2} samples across {} groups",
                residuals.len()
            ));
            true
        }
        None => true,
    }
}

/// Localize every source of a processed session.
///
/// Estimates sweep arrivals on raw takes, decides one session-wide time
/// origin (loopback, surveyed emitter, or relative-only), and converts
/// arrivals into the range matrix, then multilaterates one 3D position
/// per source row. Never fails: missing evidence yields refused cells
/// and session issues, never an aborted run.
#[expect(
    clippy::too_many_lines,
    reason = "The ordered localization pipeline keeps grouping, mode decision, and conversion together with its evidence."
)]
pub(super) fn localize_session(
    report: &ClockProcessedManifest,
    raw_root: &Path,
    output: &Path,
    fits: &[Option<CaptureClockFit>],
) -> LocalizationReport {
    let (sound_speed, sound_speed_uncertainty, sound_source) = resolve_sound_speed(&report.plan);
    let mut issues = Vec::new();
    let skeleton = |mode: LocalizationMode| LocalizationReport {
        mode,
        sound_speed_m_s: sound_speed,
        sound_speed_uncertainty_m_s: sound_speed_uncertainty,
        sound_speed_source: sound_source.into(),
        pooled_skew_ppm: None,
        mic_bias_correction_m: 0.0,
        ranges: Vec::new(),
        sources: Vec::new(),
        issues: Vec::new(),
    };
    let mut skews: Vec<f64> = fits
        .iter()
        .filter_map(|fit| fit.map(|fit| fit.skew.skew_ppm))
        .collect();
    let pooled = pooled_skew_ppm(&mut skews);
    let rate = report.plan.sample_rate_hz;
    let layout = &report.stimulus;
    let sweep: Vec<f32> = match super::clock::io::artifact(output, "stimulus.wav")
        .and_then(|path| super::clock::io::read_wave(&path, rate, layout.total_samples))
    {
        Ok(stimulus) => {
            let Some(window) =
                stimulus.get(layout.sweep_offset..layout.sweep_offset + layout.sweep_samples)
            else {
                let mut report = skeleton(LocalizationMode::RelativeOnly);
                report
                    .issues
                    .push("stimulus.wav is shorter than the journaled sweep window".into());
                return report;
            };
            window.to_vec()
        }
        Err(error) => {
            let mut report = skeleton(LocalizationMode::RelativeOnly);
            report
                .issues
                .push(format!("stimulus reference unreadable: {error}"));
            return report;
        }
    };
    let groups = select_groups(report);
    let mut arrivals: BTreeMap<usize, Result<SweepArrival, &'static str>> = BTreeMap::new();
    for group in &groups {
        for take_index in &group.take_indices {
            let arrival = take_sweep_arrival(report, output, *take_index, fits, pooled, &sweep);
            if matches!(arrival, Err("clock_outlier")) {
                let take_id = &report.takes[*take_index].raw.take_id;
                issues.push(format!(
                    "take {take_id} excluded from localization: skew outside the pooled session rate"
                ));
            }
            arrivals.insert(*take_index, arrival);
        }
    }
    let all_groups = capture_groups(report);
    let mut loopbacks: BTreeMap<(usize, u32), Result<GroupLoopback, String>> = BTreeMap::new();
    for group in &all_groups {
        loopbacks.insert(
            (group.source_index, group.repeat_index),
            group_loopback_arrival(report, raw_root, group, fits, &sweep),
        );
    }
    let nonempty = groups
        .iter()
        .filter(|group| !group.take_indices.is_empty())
        .count();
    let with_loopback = groups
        .iter()
        .filter(|group| {
            !group.take_indices.is_empty()
                && loopbacks
                    .get(&(group.source_index, group.repeat_index))
                    .is_some_and(Result::is_ok)
        })
        .count();
    for group in &groups {
        if let Some(Err(detail)) = loopbacks.get(&(group.source_index, group.repeat_index)) {
            let source = &report.plan.sources[group.source_index].id;
            issues.push(format!("loopback for source {source} unusable: {detail}"));
        }
    }
    let stable = loopback_stable(report, &all_groups, &loopbacks, fits, &mut issues);
    let mode =
        if report.plan.loopback.is_some() && with_loopback == nonempty && nonempty > 0 && stable {
            LocalizationMode::LoopbackReferenced
        } else if !groups.iter().any(|group| {
            !group.take_indices.is_empty()
                && group.take_indices.iter().any(|take| {
                    report.takes[*take].clock.basis != CaptureClockBasis::FixedAcousticReference
                })
        }) && groups.iter().any(|group| !group.take_indices.is_empty())
        {
            LocalizationMode::EmitterReferenced
        } else {
            LocalizationMode::RelativeOnly
        };
    issues.insert(
        0,
        mode_evidence(mode, nonempty, with_loopback, sound_source),
    );
    if mode == LocalizationMode::LoopbackReferenced && report.plan.timing_reference.is_some() {
        for group in &groups {
            let Some(Ok(loopback)) = loopbacks.get(&(group.source_index, group.repeat_index))
            else {
                continue;
            };
            if let Some(warning) = emitter_anchor_check(report, group, loopback, fits, &arrivals) {
                issues.push(warning);
            }
        }
    }
    let max_range = max_range_m(&report.plan);
    // A missing pool means no usable fits, so no arrival below succeeds
    // and this ratio never scales real data; 1.0 keeps refused cells exact.
    let skew_ratio = pooled.map_or(1.0, |ppm| 1.0 + ppm / 1e6);
    let mut ranges = Vec::new();
    for group in &groups {
        let row = match mode {
            LocalizationMode::LoopbackReferenced => {
                let Some(Ok(loopback)) = loopbacks.get(&(group.source_index, group.repeat_index))
                else {
                    let source = &report.plan.sources[group.source_index];
                    ranges.push(SourceRanges {
                        source_id: source.id.clone(),
                        cells: report
                            .plan
                            .microphones
                            .iter()
                            .map(|mic| refused_cell(&mic.id, "loopback_unavailable"))
                            .collect(),
                    });
                    continue;
                };
                loopback_cells(
                    report,
                    group,
                    loopback,
                    &arrivals,
                    sound_speed,
                    sound_speed_uncertainty,
                    max_range,
                    skew_ratio,
                )
            }
            LocalizationMode::EmitterReferenced => emitter_cells(
                report,
                group,
                fits,
                &arrivals,
                sound_speed,
                sound_speed_uncertainty,
                max_range,
            ),
            LocalizationMode::RelativeOnly => relative_cells(report, group, &arrivals, skew_ratio),
        };
        ranges.push(row);
    }
    let (sources, mut position_issues) =
        solve_source_positions(&report.plan, &ranges, sound_speed, rate);
    issues.append(&mut position_issues);
    LocalizationReport {
        mode,
        sound_speed_m_s: sound_speed,
        sound_speed_uncertainty_m_s: sound_speed_uncertainty,
        sound_speed_source: sound_source.into(),
        pooled_skew_ppm: pooled,
        mic_bias_correction_m: 0.0,
        ranges,
        sources,
        issues,
    }
}

/// Solve one 3D position per range row, in row order.
///
/// Rows align with plan sources via [`select_groups`]; each position looks
/// its surveyed source up by id so a future reorder cannot misattribute a
/// score. Returns the positions plus session-issue notes for outlier
/// exclusions and refused fits.
fn solve_source_positions(
    plan: &CaptureSessionPlan,
    ranges: &[SourceRanges],
    sound_speed_m_s: f64,
    sample_rate_hz: u32,
) -> (Vec<SourcePosition>, Vec<String>) {
    let sample_m = sound_speed_m_s / f64::from(sample_rate_hz.max(1));
    let mut sources = Vec::with_capacity(ranges.len());
    let mut issues = Vec::new();
    for row in ranges {
        let declared = plan
            .sources
            .iter()
            .find(|source| source.id == row.source_id)
            .and_then(|source| source.position_m);
        let (position, mut notes) = solve_row_position(row, &plan.microphones, declared, sample_m);
        sources.push(position);
        issues.append(&mut notes);
    }
    (sources, issues)
}

/// Multilaterate one source row with geometry and residual gates.
///
/// Cells pair with plan microphones by order (the builders' contract) and
/// by id; a misaligned cell counts as missing rather than misattributed.
/// Usable cells mirror the solver's validity rules exactly, so the
/// geometry pre-check and the solver classify the same subset: `None`
/// from the solver then means depletion or divergence (`solve_failed`),
/// never a disagreement about the input. Refusals set `reason` and leave
/// the position empty; `solve_residual_high` still publishes the RMS and
/// GDOP of the refused fit as evidence.
fn solve_row_position(
    row: &SourceRanges,
    microphones: &[CaptureMicrophone],
    declared_m: Option<[f64; 3]>,
    sample_m: f64,
) -> (SourcePosition, Vec<String>) {
    let blank = |reason: Option<&str>, mirror_ambiguous: bool| SourcePosition {
        source_id: row.source_id.clone(),
        position_m: None,
        sigma_m: None,
        rms_residual_m: None,
        gdop: None,
        mirror_ambiguous,
        declared_error_m: None,
        reason: reason.map(str::to_string),
    };
    let mut notes = Vec::new();
    let mut usable: Vec<(usize, [f64; 3], f64, f64)> = Vec::new();
    for (index, cell) in row.cells.iter().enumerate() {
        let Some(microphone) = microphones.get(index) else {
            continue;
        };
        if cell.microphone_id != microphone.id {
            continue;
        }
        if let (Some(distance), Some(sigma)) = (cell.distance_m, cell.sigma_m) {
            if distance.is_finite() && distance >= 0.0 && sigma.is_finite() && sigma > 0.0 {
                usable.push((index, microphone.position_m, distance, sigma));
            }
        }
    }
    if usable.len() < MIN_RANGES_FOR_FIX {
        if usable.len() == 3 {
            let positions: Vec<[f64; 3]> =
                usable.iter().map(|(_, position, _, _)| *position).collect();
            if geometry_rank(&positions).collinear {
                return (blank(Some("degenerate_geometry"), false), notes);
            }
            return (blank(Some("mirror_ambiguous"), true), notes);
        }
        return (blank(Some("insufficient_ranges"), false), notes);
    }
    let positions: Vec<[f64; 3]> = usable.iter().map(|(_, position, _, _)| *position).collect();
    if geometry_rank(&positions).collinear {
        return (blank(Some("degenerate_geometry"), false), notes);
    }
    let ranges: Vec<Option<(f64, f64)>> = usable
        .iter()
        .map(|(_, _, distance, sigma)| Some((*distance, *sigma)))
        .collect();
    let Some(estimate) = solve_position(&positions, &ranges) else {
        notes.push(format!(
            "source {} position refused: multilateration failed (solve_failed)",
            row.source_id
        ));
        return (blank(Some("solve_failed"), false), notes);
    };
    for (slot, used) in estimate.used.iter().enumerate() {
        if !used {
            let microphone_id = &row.cells[usable[slot].0].microphone_id;
            notes.push(format!(
                "source {}: {microphone_id} excluded from the position fit as a range outlier",
                row.source_id
            ));
        }
    }
    let mut used_sigmas: Vec<f64> = estimate
        .used
        .iter()
        .enumerate()
        .filter(|(_, used)| **used)
        .map(|(slot, _)| usable[slot].3)
        .collect();
    used_sigmas.sort_by(f64::total_cmp);
    let median_sigma = median_sorted(&used_sigmas).unwrap_or(0.0);
    // Q7: refuse fits whose RMS residual exceeds one sample or three
    // median cell sigmas. `max` falls back to the sigma term when the
    // sample term is degenerate, so the gate stays fail-closed.
    let residual_gate_m = sample_m.max(3.0 * median_sigma);
    if estimate.rms_residual_m > residual_gate_m {
        notes.push(format!(
            "source {} position refused: RMS residual {:.1} mm exceeds the {:.1} mm gate (solve_residual_high)",
            row.source_id,
            estimate.rms_residual_m * 1000.0,
            residual_gate_m * 1000.0,
        ));
        let mut refused = blank(Some("solve_residual_high"), estimate.mirror_ambiguous);
        refused.rms_residual_m = Some(estimate.rms_residual_m);
        refused.gdop = Some(estimate.gdop);
        return (refused, notes);
    }
    let declared_error_m = declared_m.map(|surveyed| euclidean_m(surveyed, estimate.position_m));
    let position = SourcePosition {
        source_id: row.source_id.clone(),
        position_m: Some(estimate.position_m),
        sigma_m: Some(estimate.sigma_m),
        rms_residual_m: Some(estimate.rms_residual_m),
        gdop: Some(estimate.gdop),
        mirror_ambiguous: estimate.mirror_ambiguous,
        declared_error_m,
        reason: None,
    };
    (position, notes)
}

/// True median of an ascending slice; `None` when empty.
fn median_sorted(sorted: &[f64]) -> Option<f64> {
    let count = sorted.len();
    if count == 0 {
        return None;
    }
    if count % 2 == 1 {
        return Some(sorted[count / 2]);
    }
    Some(sorted[count / 2 - 1].midpoint(sorted[count / 2]))
}

/// Euclidean distance between two plan-frame points in meters.
fn euclidean_m(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Localization readiness warnings for `plan-validate`, in print order.
///
/// Names the expected time-origin mode from the declared references, then
/// flags microphone geometry that cannot produce 3D positions. The actual
/// mode can still degrade at process time when a reference proves unusable.
#[must_use]
pub fn localization_readiness(plan: &CaptureSessionPlan) -> Vec<String> {
    let mut lines = Vec::new();
    let microphones = plan.microphones.len();
    let positions: Vec<[f64; 3]> = plan
        .microphones
        .iter()
        .map(|microphone| microphone.position_m)
        .collect();
    let rank = geometry_rank(&positions);
    let mode = if plan.loopback.is_some() {
        "loopback-referenced (electrical loopback wired)"
    } else if plan.timing_reference.is_some() {
        "emitter-referenced (surveyed timing emitter)"
    } else {
        "relative-only (no absolute time origin; wire the loopback or survey the timing emitter)"
    };
    lines.push(format!(
        "localization readiness: {microphones} microphones, expecting {mode}"
    ));
    if microphones < MIN_RANGES_FOR_FIX {
        lines.push(format!(
            "localization readiness: {microphones}/4 microphones, positions need at least 4"
        ));
    }
    if rank.collinear {
        lines.push(
            "localization readiness: microphone geometry is collinear, 3D positions impossible"
                .into(),
        );
    } else if rank.coplanar {
        lines.push(
            "localization readiness: microphones are coplanar, positions stay mirror-ambiguous without height diversity"
                .into(),
        );
    }
    lines
}

/// One CLI line per localization outcome, in plan source order.
///
/// The first line names the session mode and sound-speed provenance;
/// each following line summarizes one source row, with a position line
/// after each row that carries a position entry. Session issues print
/// separately as warnings by the caller.
#[must_use]
pub fn localization_summary_lines(report: &LocalizationReport) -> Vec<String> {
    let mut lines = vec![format!(
        "localization mode: {} (sound speed {:.1} m/s from {})",
        report.mode, report.sound_speed_m_s, report.sound_speed_source
    )];
    for row in &report.ranges {
        let total = row.cells.len();
        if report.mode == LocalizationMode::RelativeOnly {
            let delays = row
                .cells
                .iter()
                .filter(|cell| cell.tdoa_samples_vs_first.is_some())
                .count();
            let source_id = &row.source_id;
            lines.push(format!(
                "localization {source_id}: relative delays for {delays}/{total} microphones"
            ));
        } else {
            let mut distances: Vec<f64> = row
                .cells
                .iter()
                .filter_map(|cell| cell.distance_m)
                .collect();
            distances.sort_by(f64::total_cmp);
            let mut reasons: Vec<&str> = row
                .cells
                .iter()
                .filter(|cell| cell.distance_m.is_none())
                .filter_map(|cell| cell.reason.as_deref())
                .collect();
            reasons.sort_unstable();
            reasons.dedup();
            let source_id = &row.source_id;
            let ranged = distances.len();
            let span = if let (Some(first), Some(last)) = (distances.first(), distances.last()) {
                format!(" {first:.2}–{last:.2} m")
            } else {
                String::new()
            };
            let refused = if reasons.is_empty() {
                String::new()
            } else {
                let reasons = reasons.join(", ");
                format!(" (refused: {reasons})")
            };
            lines.push(format!(
                "localization {source_id}: {ranged}/{total} ranges{span}{refused}"
            ));
        }
        if let Some(position) = report
            .sources
            .iter()
            .find(|source| source.source_id == row.source_id)
        {
            lines.push(position_summary_line(position));
        }
    }
    lines
}

/// One CLI line for a solved or refused source position.
fn position_summary_line(position: &SourcePosition) -> String {
    let source_id = &position.source_id;
    let mut detail = String::new();
    if let Some(rms) = position.rms_residual_m {
        let _ = write!(detail, " rms {:.1} mm", rms * 1000.0);
    }
    if let Some(gdop) = position.gdop {
        let _ = write!(detail, " gdop {gdop:.1}");
    }
    if position.mirror_ambiguous {
        detail.push_str(" [mirror-ambiguous]");
    }
    if let Some([x, y, z]) = position.position_m {
        format!("localization {source_id} position: [{x:.3}, {y:.3}, {z:.3}] m{detail}")
    } else {
        let reason = position.reason.as_deref().unwrap_or("unknown");
        format!("localization {source_id} position: refused ({reason}){detail}")
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "Synthetic fixtures use bounded sample counts and intentionally quantized audio; exact comparisons verify unchanged settings and samples."
)]
mod tests {
    use super::*;
    use crate::capture_session::protocol::prepare_capture_stimulus;
    use crate::capture_session::record::{RawCaptureManifest, RawCaptureStatus};
    use crate::capture_session::{CaptureLoopback, CaptureSoundSpeed, CaptureTimingReference};

    #[test]
    fn sound_speed_prefers_emitter_then_plan_then_default() {
        let mut plan: CaptureSessionPlan =
            serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json"))
                .unwrap();
        assert_eq!(resolve_sound_speed(&plan).2, "default");
        assert_eq!(resolve_sound_speed(&plan).0, 343.0);
        plan.sound_speed = Some(CaptureSoundSpeed {
            m_s: 340.0,
            uncertainty_m_s: 2.0,
            air_temperature_c: Some(15.0),
        });
        assert_eq!(resolve_sound_speed(&plan), (340.0, 2.0, "plan"));
        plan.timing_reference = Some(CaptureTimingReference {
            output_channel: 2,
            position_m: [0.0, 0.0, 0.0],
            position_uncertainty_mm: 1.0,
            sound_speed_m_s: 345.0,
            sound_speed_uncertainty_m_s: 1.0,
        });
        assert_eq!(resolve_sound_speed(&plan).2, "timing_reference");
    }

    #[test]
    fn pooled_skew_is_the_upper_middle_and_rejects_nonfinite() {
        assert_eq!(pooled_skew_ppm(&mut []), None);
        assert_eq!(pooled_skew_ppm(&mut [3.0]), Some(3.0));
        assert_eq!(pooled_skew_ppm(&mut [1.0, 2.0, 9.0]), Some(2.0));
        assert_eq!(pooled_skew_ppm(&mut [1.0, 2.0, 3.0, 9.0]), Some(3.0));
        assert_eq!(pooled_skew_ppm(&mut [1.0, f64::NAN]), None);
        assert_eq!(pooled_skew_ppm(&mut [f64::INFINITY]), None);
    }

    #[test]
    fn arrival_sigma_floors_clean_inputs_and_admits_unknown() {
        let clean = arrival_sigma_samples(100.0, 20.0, 20_000.0, 48_000.0);
        assert!((clean - ARRIVAL_SIGMA_FLOOR_SAMPLES).abs() < 1e-12);
        let infinite = arrival_sigma_samples(f64::INFINITY, 20.0, 20_000.0, 48_000.0);
        assert!((infinite - ARRIVAL_SIGMA_FLOOR_SAMPLES).abs() < 1e-12);
        let noisy = arrival_sigma_samples(6.0, 20.0, 20_000.0, 48_000.0);
        assert!(noisy.is_finite() && noisy > clean);
        assert_eq!(
            arrival_sigma_samples(f64::NAN, 20.0, 20_000.0, 48_000.0),
            f64::INFINITY
        );
        assert_eq!(
            arrival_sigma_samples(30.0, 8000.0, 2000.0, 48_000.0),
            f64::INFINITY
        );
    }

    #[test]
    fn range_bounds_and_conversions_follow_the_plan() {
        let mut plan: CaptureSessionPlan =
            serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json"))
                .unwrap();
        assert_eq!(max_range_m(&plan), DEFAULT_MAX_RANGE_M);
        plan.room_dimensions = Some(crate::capture_session::RoomDimensions {
            length: 3.0,
            width: 4.0,
            height: 12.0,
        });
        assert!((max_range_m(&plan) - (13.0 + RANGE_SLACK_M)).abs() < 1e-9);
        let source = &plan.sources[0];
        assert!(
            (propagation_to_meters(480.0, 48_000.0, 343.0, source).unwrap() - 3.43).abs() < 1e-9
        );
        let delayed = crate::capture_session::CaptureSource {
            source_delay_calibration_ms: Some(1.0),
            ..source.clone()
        };
        assert!(
            (propagation_to_meters(480.0, 48_000.0, 343.0, &delayed).unwrap() - (3.43 - 0.343))
                .abs()
                < 1e-9
        );
        assert_eq!(
            propagation_to_meters(f64::NAN, 48_000.0, 343.0, source),
            None
        );
        let sigma = range_sigma_m(1.0, 3.43, 48_000.0, 343.0, 0.5).unwrap();
        assert!(sigma > 343.0 / 48_000.0 && sigma < 0.05);
        assert_eq!(
            range_sigma_m(f64::INFINITY, 3.43, 48_000.0, 343.0, 0.5),
            None
        );
    }

    /// Tapered log sweep matching production playback conditioning.
    ///
    /// The raw generator has no edge fade; abrupt endpoints splatter
    /// correlation sidelobes far above the direct picker's threshold.
    /// Production stimuli fade both edges, so the fixture does too.
    fn sweep_fixture() -> Vec<f32> {
        let mut sweep = math_audio_dsp::signals::gen_log_sweep(20.0, 20_000.0, 0.5, 48_000, 0.25);
        let len = sweep.len();
        let fade = (48_000_usize * 10 / 1000).min(len / 4);
        for (index, sample) in sweep.iter_mut().enumerate() {
            let edge = index.min(len - 1 - index).min(fade);
            let gain = 0.5 - 0.5 * (std::f32::consts::PI * edge as f32 / fade as f32).cos();
            *sample *= gain;
        }
        sweep
    }

    /// Hann-windowed sinc interpolation for the synthesis oracle.
    ///
    /// Linear interpolation distorts near-Nyquist sweep content enough to
    /// bias arrival estimates, so the oracle resamples with a symmetric
    /// 17-tap windowed sinc instead (even tap counts center half a sample
    /// early and bias every arrival). Zero-padded past the signal edges.
    fn sinc_at(signal: &[f32], time: f64) -> f32 {
        const TAPS: i32 = 17;
        const HALF: i32 = TAPS / 2;
        if time < 0.0 || time >= signal.len() as f64 - 1.0 {
            return 0.0;
        }
        let center = time.floor() as i32;
        let mut acc = 0.0;
        for tap in -HALF..=HALF {
            let index = center + tap;
            let sample = if index < 0 || index >= signal.len() as i32 {
                0.0
            } else {
                f64::from(signal[index as usize])
            };
            let offset = time - f64::from(center + tap);
            let sinc = if offset.abs() < 1e-9 {
                1.0
            } else {
                let phase = std::f64::consts::PI * offset;
                phase.sin() / phase
            };
            let window = 0.5
                - 0.5
                    * (2.0 * std::f64::consts::PI * f64::from(tap + HALF) / f64::from(TAPS - 1))
                        .cos();
            acc += sample * sinc * window;
        }
        acc as f32
    }

    fn delayed(signal: &[f32], delay_samples: f64, extra: usize) -> Vec<f32> {
        (0..signal.len() + extra)
            .map(|n| sinc_at(signal, n as f64 - delay_samples))
            .collect()
    }

    /// Flat-spectrum linear chirp isolating estimator bias from sweep slope.
    fn chirp_fixture() -> Vec<f32> {
        let rate = 48_000.0;
        let len = (rate * 0.25) as usize;
        (0..len)
            .map(|n| {
                let t = n as f64 / rate;
                let phase = 2.0 * std::f64::consts::PI * (2000.0 * t + (6000.0 * t * t) / 0.5);
                (phase.sin() * 0.5) as f32
            })
            .collect()
    }

    #[test]
    fn estimator_recovers_fractional_delays() {
        for (name, sweep) in [("sweep", sweep_fixture()), ("chirp", chirp_fixture())] {
            for delay in [37.25, 500.0, 1000.5] {
                let recorded = delayed(&sweep, delay, 4096);
                let arrival =
                    estimate_sweep_arrival(&sweep, &recorded, delay, 48_000, 20.0, 20_000.0)
                        .unwrap();
                assert!(
                    (arrival.direct_samples - delay).abs() < 0.2,
                    "{name} delay {delay}: got {}",
                    arrival.direct_samples
                );
                assert!(arrival.confidence_db >= MIN_TDOA_CONFIDENCE_DB);
                assert!(arrival.sigma_samples.is_finite());
            }
        }
    }

    #[test]
    fn estimator_keeps_a_weaker_reflection_but_flags_a_stronger_late_one() {
        let sweep = sweep_fixture();
        let mut recorded = delayed(&sweep, 500.0, 4096);
        let echo = delayed(&sweep, 508.0, 4096);
        for (sample, stray) in recorded.iter_mut().zip(echo.iter()) {
            *sample += stray * 0.5;
        }
        let arrival =
            estimate_sweep_arrival(&sweep, &recorded, 500.0, 48_000, 20.0, 20_000.0).unwrap();
        assert!((arrival.direct_samples - 500.0).abs() < 0.5);
        let mut recorded = delayed(&sweep, 500.0, 4096);
        let echo = delayed(&sweep, 700.0, 4096);
        for (sample, stray) in recorded.iter_mut().zip(echo.iter()) {
            *sample += stray * 2.0;
        }
        assert_eq!(
            estimate_sweep_arrival(&sweep, &recorded, 500.0, 48_000, 20.0, 20_000.0).unwrap_err(),
            "arrival_ambiguous"
        );
    }

    #[test]
    fn estimator_refuses_silence_and_missing_windows() {
        let sweep = sweep_fixture();
        let silent = vec![0.0; sweep.len() + 4096];
        assert_eq!(
            estimate_sweep_arrival(&sweep, &silent, 0.0, 48_000, 20.0, 20_000.0).unwrap_err(),
            "arrival_low_confidence"
        );
        assert_eq!(
            estimate_sweep_arrival(&sweep, &sweep, f64::NAN, 48_000, 20.0, 20_000.0).unwrap_err(),
            "sweep_window_missing"
        );
        assert_eq!(
            estimate_sweep_arrival(&[], &silent, 0.0, 48_000, 20.0, 20_000.0).unwrap_err(),
            "sweep_window_missing"
        );
    }

    #[derive(serde::Deserialize)]
    struct GoldenClock {
        offset_samples: f64,
        skew_ppm: f64,
    }

    #[derive(serde::Deserialize)]
    struct Golden {
        sample_rate_hz: u32,
        sound_speed_m_s: f64,
        sound_speed_uncertainty_m_s: f64,
        session_clock: GoldenClock,
        latency_samples: f64,
        microphones_m: BTreeMap<String, [f64; 3]>,
        sources_m: BTreeMap<String, [f64; 3]>,
        timing_emitter_m: [f64; 3],
        expected_ranges_m: BTreeMap<String, f64>,
    }

    fn golden() -> Golden {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/localization-golden.json"
        ))
        .unwrap()
    }

    fn distance_meter(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum HarnessMode {
        Loopback,
        Emitter,
        Relative,
    }

    /// Synthesize one take through separate sweep and chirp paths.
    ///
    /// The sweep travels the source path (latency plus source
    /// propagation); the timing chirps travel the chirp path (latency
    /// plus emitter propagation, or the source path when no dedicated
    /// emitter exists). Windowed-sinc resampling keeps full-band sweep
    /// content intact, unlike the linear interpolation the narrower
    /// chirp-only clock tests can afford.
    fn synth_take(
        stimulus: &[f32],
        layout: &crate::capture_session::protocol::CaptureStimulusLayout,
        offset: f64,
        skew_ppm: f64,
        latency: f64,
        sweep_prop: f64,
        chirp_prop: f64,
    ) -> Vec<f32> {
        let in_chirp = |index: usize| {
            (layout.start_chirp_offset..layout.start_chirp_offset + layout.chirp_samples)
                .contains(&index)
                || (layout.end_chirp_offset..layout.end_chirp_offset + layout.chirp_samples)
                    .contains(&index)
        };
        let sweep_only: Vec<f32> = stimulus
            .iter()
            .enumerate()
            .map(|(index, sample)| if in_chirp(index) { 0.0 } else { *sample })
            .collect();
        let chirps_only: Vec<f32> = stimulus
            .iter()
            .enumerate()
            .map(|(index, sample)| if in_chirp(index) { *sample } else { 0.0 })
            .collect();
        let ratio = 1.0 + skew_ppm / 1e6;
        (0..stimulus.len() + 4096)
            .map(|n| {
                let base = (n as f64 - offset) / ratio - latency;
                sinc_at(&sweep_only, base - sweep_prop) + sinc_at(&chirps_only, base - chirp_prop)
            })
            .collect()
    }

    struct Harness {
        report: crate::capture_session::clock::io::ClockProcessedManifest,
        fits: Vec<Option<crate::capture_session::clock::CaptureClockFit>>,
        raw_root: tempfile::TempDir,
        output: tempfile::TempDir,
        golden: Golden,
    }

    /// Build a complete synthetic session for one localization mode.
    ///
    /// Takes and loopbacks are synthesized in memory from the golden
    /// geometry, fitted with the real clock estimator, and laid out on
    /// disk exactly as `plan-process` leaves them, so `localize_session`
    /// runs its full file-backed path.
    #[expect(
        clippy::too_many_lines,
        reason = "The golden harness keeps plan, synthesis, fitting, and layout in one explicit sequence."
    )]
    fn harness(mode: HarnessMode) -> Harness {
        let golden = golden();
        let rate = golden.sample_rate_hz;
        let mut plan: CaptureSessionPlan =
            serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json"))
                .unwrap();
        plan.sweep.duration_secs = 0.25;
        let microphone_template = plan.microphones[0].clone();
        plan.microphones = golden
            .microphones_m
            .iter()
            .enumerate()
            .map(|(channel, (id, position))| {
                let mut mic = microphone_template.clone();
                mic.id = id.clone();
                mic.device = "iface".into();
                mic.input_channel = channel as u16;
                mic.position_m = *position;
                mic.position_uncertainty_mm = 0.1;
                mic
            })
            .collect();
        plan.sources = golden
            .sources_m
            .keys()
            .enumerate()
            .map(|(channel, id)| crate::capture_session::CaptureSource {
                id: id.clone(),
                output_channel: channel as u16,
                position_m: None,
                position_uncertainty_mm: None,
                source_delay_calibration_ms: None,
            })
            .collect();
        plan.sound_speed = Some(CaptureSoundSpeed {
            m_s: golden.sound_speed_m_s,
            uncertainty_m_s: golden.sound_speed_uncertainty_m_s,
            air_temperature_c: None,
        });
        if mode == HarnessMode::Emitter {
            plan.timing_reference = Some(CaptureTimingReference {
                output_channel: 2,
                position_m: golden.timing_emitter_m,
                position_uncertainty_mm: 0.1,
                sound_speed_m_s: golden.sound_speed_m_s,
                sound_speed_uncertainty_m_s: 0.1,
            });
        }
        if mode == HarnessMode::Loopback {
            plan.loopback = Some(CaptureLoopback {
                device: "iface".into(),
                input_channel: 7,
            });
        }
        let session = plan.clone().validate().unwrap();
        let stimulus = prepare_capture_stimulus(&session).unwrap();
        let layout = stimulus.layout.clone();
        let rate_f64 = f64::from(rate);
        let raw_root = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let mut manifest = RawCaptureManifest {
            version: 1,
            plan: plan.clone(),
            loopback_files: Vec::new(),
            probe_pass: None,
            bass_anchor_pass: None,
            stimulus: layout.clone(),
            timing_reference_output_channel: Some(if mode == HarnessMode::Emitter { 2 } else { 0 }),
            status: RawCaptureStatus::RawComplete,
            pending_processing: Vec::new(),
            calibrations: Vec::new(),
            takes: Vec::new(),
            error: None,
        };
        let mut takes = Vec::new();
        let mut fits = Vec::new();
        let mut audio_index = 0;
        // Without a dedicated emitter the chirps ride the first source
        // channel for every take, matching the record path's rerouting.
        let chirp_source = golden
            .sources_m
            .values()
            .next()
            .copied()
            .unwrap_or([0.0; 3]);
        for (source_id, source_position) in &golden.sources_m {
            for (mic_id, mic_position) in &golden.microphones_m {
                let src_prop = distance_meter(*source_position, *mic_position)
                    / golden.sound_speed_m_s
                    * rate_f64;
                let chirp_prop = if mode == HarnessMode::Emitter {
                    distance_meter(golden.timing_emitter_m, *mic_position) / golden.sound_speed_m_s
                        * rate_f64
                } else {
                    distance_meter(chirp_source, *mic_position) / golden.sound_speed_m_s * rate_f64
                };
                let samples = synth_take(
                    &stimulus.samples,
                    &layout,
                    golden.session_clock.offset_samples,
                    golden.session_clock.skew_ppm,
                    golden.latency_samples,
                    src_prop,
                    chirp_prop,
                );
                let wav_file = format!("take-{source_id}-{mic_id}-raw.wav");
                crate::signal_recorder::write_wav_file(
                    &output.path().join(&wav_file),
                    &samples,
                    rate,
                    1,
                )
                .unwrap();
                let take = crate::capture_session::record::RawCaptureTake {
                    take_id: crate::capture_session::record::stable_take_id(source_id, mic_id, 0),
                    repeat_index: 0,
                    device_id: "id:iface".into(),
                    output_device_id: "id:iface".into(),
                    source_id: source_id.clone(),
                    microphone_id: mic_id.clone(),
                    wav_file,
                    samples: samples.len(),
                    input_sample_format: "F32".into(),
                    output_sample_format: "F32".into(),
                    peak_amplitude: samples
                        .iter()
                        .fold(0.0_f32, |peak, sample| peak.max(sample.abs())),
                    clipped_samples: 0,
                };
                let (provenance, fit) = crate::capture_session::clock::fit_capture_clock(
                    &manifest,
                    &take,
                    &samples,
                    &stimulus.timing_chirp,
                );
                let corrected = crate::capture_session::clock::apply_clock_fit(
                    provenance,
                    fit,
                    &samples,
                    layout.total_samples,
                );
                assert!(
                    fit.is_some(),
                    "synthetic take {source_id}/{mic_id} must fit"
                );
                assert!(corrected.samples.is_some());
                let analysis = crate::capture_session::analysis::CaptureAnalysisReport {
                    magnitude_file: None,
                    clipped_samples: 0,
                    broadband_snr_db: Some(40.0),
                    frequency_snr: Vec::new(),
                    common_reference: None,
                    issues: Vec::new(),
                    quality_findings: Vec::new(),
                };
                takes.push(crate::capture_session::clock::io::ClockProcessedTake {
                    raw: take.clone(),
                    audio_file: format!("take-{audio_index:03}.wav"),
                    clock: corrected.provenance,
                    analysis: Some(analysis),
                });
                manifest.takes.push(take);
                fits.push(fit);
                audio_index += 1;
            }
            if mode == HarnessMode::Loopback {
                let samples = synth_take(
                    &stimulus.samples,
                    &layout,
                    golden.session_clock.offset_samples,
                    golden.session_clock.skew_ppm,
                    golden.latency_samples,
                    0.0,
                    0.0,
                );
                let wav_file = format!("take-{source_id}-loopback-raw.wav");
                crate::signal_recorder::write_wav_file(
                    &raw_root.path().join(&wav_file),
                    &samples,
                    rate,
                    1,
                )
                .unwrap();
                manifest
                    .loopback_files
                    .push(crate::capture_session::record::CaptureLoopbackTake {
                        repeat_index: 0,
                        source_id: source_id.clone(),
                        wav_file,
                        samples: samples.len(),
                        device_id: "id:iface".into(),
                        input_sample_format: "F32".into(),
                        peak_amplitude: samples
                            .iter()
                            .fold(0.0_f32, |peak, sample| peak.max(sample.abs())),
                        clipped_samples: 0,
                    });
            }
        }
        crate::signal_recorder::write_wav_file(
            &output.path().join("stimulus.wav"),
            &stimulus.samples,
            rate,
            1,
        )
        .unwrap();
        let report = crate::capture_session::clock::io::ClockProcessedManifest {
            version: 1,
            plan,
            raw_status: Some(RawCaptureStatus::RawComplete),
            selected_take_ids: None,
            ir_microphone: None,
            timing_reference_output_channel: manifest.timing_reference_output_channel,
            parent_inventory_file: None,
            stimulus: layout,
            calibrations: Vec::new(),
            takes,
            pending_processing: Vec::new(),
            probe_pass: None,
            bass_anchor_pass: None,
            loopback_files: manifest.loopback_files.clone(),
            ctc_measurements: None,
            ctc_files: Vec::new(),
            recording_manifest: None,
            reflection_reports: Vec::new(),
            localization_report: None,
        };
        Harness {
            report,
            fits,
            raw_root,
            output,
            golden,
        }
    }

    #[test]
    fn loopback_session_recovers_golden_ranges() {
        let harness = harness(HarnessMode::Loopback);
        let localized = localize_session(
            &harness.report,
            harness.raw_root.path(),
            harness.output.path(),
            &harness.fits,
        );
        assert_eq!(localized.mode, LocalizationMode::LoopbackReferenced);
        assert_eq!(localized.sound_speed_source, "plan");
        assert_eq!(localized.ranges.len(), 2);
        for row in &localized.ranges {
            assert_eq!(row.cells.len(), 4);
            for cell in &row.cells {
                let key = format!("{}/{}", row.source_id, cell.microphone_id);
                let expected = harness.golden.expected_ranges_m[&key];
                let distance = cell.distance_m.unwrap_or(f64::NAN);
                assert!(
                    (distance - expected).abs() < 0.002,
                    "{key}: got {distance}, want {expected}"
                );
                let sigma = cell.sigma_m.expect("range needs uncertainty");
                assert!(sigma.is_finite() && sigma > 0.0 && sigma < 0.05);
                assert!(cell.arrival_samples.is_some());
                assert!(cell.reference_arrival_samples.is_some());
                assert!(cell.confidence_db.is_some());
                assert_eq!(cell.reason, None);
                assert_eq!(cell.band.as_deref(), Some("sweep_2000_8000"));
            }
        }
        assert!(
            (localized.pooled_skew_ppm.unwrap() - harness.golden.session_clock.skew_ppm).abs()
                < 0.5
        );
        let lines = localization_summary_lines(&localized);
        assert!(lines[0].contains("loopback-referenced"));
        assert!(lines[1].contains("4/4 ranges"));
        assert_eq!(localized.sources.len(), 2);
        for position in &localized.sources {
            let truth = harness.golden.sources_m[&position.source_id];
            let fit = position
                .position_m
                .expect("loopback ranges solve positions");
            let error = distance_meter(fit, truth);
            assert!(
                error < 0.02,
                "{}: position error {error} m",
                position.source_id
            );
            assert!(!position.mirror_ambiguous);
            assert_eq!(position.reason, None);
            assert_eq!(position.declared_error_m, None);
            assert!(position.rms_residual_m.unwrap_or(f64::NAN) < 0.01);
            assert!(position.gdop.unwrap_or(f64::NAN).is_finite());
            assert!(position.sigma_m.unwrap().iter().all(|sigma| *sigma > 0.0));
        }
        assert!(lines[2].contains("position: ["));
    }

    #[test]
    fn emitter_session_recovers_golden_ranges() {
        let harness = harness(HarnessMode::Emitter);
        for take in &harness.report.takes {
            assert_eq!(
                take.clock.basis,
                crate::capture_session::clock::CaptureClockBasis::FixedAcousticReference
            );
        }
        let localized = localize_session(
            &harness.report,
            harness.raw_root.path(),
            harness.output.path(),
            &harness.fits,
        );
        assert_eq!(localized.mode, LocalizationMode::EmitterReferenced);
        assert_eq!(localized.sound_speed_source, "timing_reference");
        for row in &localized.ranges {
            for cell in &row.cells {
                let key = format!("{}/{}", row.source_id, cell.microphone_id);
                let expected = harness.golden.expected_ranges_m[&key];
                let distance = cell.distance_m.unwrap_or(f64::NAN);
                assert!(
                    (distance - expected).abs() < 0.002,
                    "{key}: got {distance}, want {expected}"
                );
                assert!(cell.sigma_m.is_some());
                assert_eq!(cell.reason, None);
            }
        }
    }

    #[test]
    fn relative_session_reports_delays_without_distances() {
        let harness = harness(HarnessMode::Relative);
        let localized = localize_session(
            &harness.report,
            harness.raw_root.path(),
            harness.output.path(),
            &harness.fits,
        );
        assert_eq!(localized.mode, LocalizationMode::RelativeOnly);
        let rate = f64::from(harness.golden.sample_rate_hz);
        for row in &localized.ranges {
            let source = harness.golden.sources_m[&row.source_id];
            let first = harness.golden.microphones_m["mic-0"];
            for cell in &row.cells {
                assert_eq!(cell.distance_m, None);
                assert_eq!(cell.reason.as_deref(), Some("no_absolute_time_origin"));
                let microphone = harness.golden.microphones_m[&cell.microphone_id];
                let expected = (distance_meter(source, microphone) - distance_meter(source, first))
                    / harness.golden.sound_speed_m_s
                    * rate;
                let delay = cell.tdoa_samples_vs_first.unwrap_or(f64::NAN);
                assert!(
                    (delay - expected).abs() < 0.2,
                    "{}: got {delay}, want {expected}",
                    cell.microphone_id
                );
            }
        }
        let lines = localization_summary_lines(&localized);
        assert!(lines[1].contains("relative delays for 4/4"));
    }

    /// Plan with controlled microphone geometry and one source, for
    /// position-gate tests without audio synthesis.
    fn position_plan(
        microphones: &[(&str, [f64; 3])],
        declared_m: Option<[f64; 3]>,
    ) -> CaptureSessionPlan {
        let mut plan: CaptureSessionPlan =
            serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json"))
                .unwrap();
        let template = plan.microphones[0].clone();
        plan.microphones = microphones
            .iter()
            .enumerate()
            .map(|(channel, (id, position))| {
                let mut microphone = template.clone();
                microphone.id = (*id).to_string();
                microphone.input_channel = channel as u16;
                microphone.position_m = *position;
                microphone
            })
            .collect();
        plan.sources = vec![crate::capture_session::CaptureSource {
            id: "left".into(),
            output_channel: 0,
            position_m: declared_m,
            position_uncertainty_mm: declared_m.map(|_| 1.0),
            source_delay_calibration_ms: None,
        }];
        plan
    }

    /// One range row from `(microphone id, range)` pairs; `None` ranges
    /// model refused cells.
    fn position_row(source_id: &str, cells: Vec<(&str, Option<(f64, f64)>)>) -> SourceRanges {
        SourceRanges {
            source_id: source_id.into(),
            cells: cells
                .into_iter()
                .map(|(microphone_id, range)| {
                    let (distance_m, sigma_m) = match range {
                        Some((distance, sigma)) => (Some(distance), Some(sigma)),
                        None => (None, None),
                    };
                    RangeCell {
                        microphone_id: microphone_id.into(),
                        distance_m,
                        sigma_m,
                        arrival_samples: None,
                        reference_arrival_samples: None,
                        tdoa_samples_vs_first: None,
                        tdoa_sigma_samples: None,
                        band: None,
                        confidence_db: None,
                        reason: if distance_m.is_none() {
                            Some("test_refused".into())
                        } else {
                            None
                        },
                    }
                })
                .collect(),
        }
    }

    /// Exact ranges from a truth position over tetra microphone geometry.
    fn tetra_row(truth: [f64; 3], sigma_m: f64) -> (Vec<(&'static str, [f64; 3])>, SourceRanges) {
        let microphones = vec![
            ("mic-0", [0.0, 0.0, 0.0]),
            ("mic-1", [1.0, 0.0, 0.0]),
            ("mic-2", [0.0, 1.0, 0.0]),
            ("mic-3", [0.0, 0.0, 1.0]),
        ];
        let row = position_row(
            "left",
            microphones
                .iter()
                .map(|(id, position)| (*id, Some((distance_meter(*position, truth), sigma_m))))
                .collect(),
        );
        (microphones, row)
    }

    #[test]
    fn median_sorted_picks_the_true_middle() {
        assert_eq!(median_sorted(&[]), None);
        assert_eq!(median_sorted(&[2.0]), Some(2.0));
        assert_eq!(median_sorted(&[1.0, 2.0, 9.0]), Some(2.0));
        assert_eq!(median_sorted(&[1.0, 2.0, 3.0, 9.0]), Some(2.5));
    }

    #[test]
    fn positions_solve_and_score_surveyed_sources() {
        let truth = [2.3, -1.1, 0.7];
        let (microphones, row) = tetra_row(truth, 0.001);
        let plan = position_plan(&microphones, Some(truth));
        let (sources, issues) = solve_source_positions(&plan, &[row], 343.0, 48_000);
        assert_eq!(issues, Vec::<String>::new());
        let position = &sources[0];
        let fit = position.position_m.expect("tetra ranges solve");
        assert!(distance_meter(fit, truth) < 1.0e-6);
        assert_eq!(position.reason, None);
        assert!(!position.mirror_ambiguous);
        assert!(position.declared_error_m.unwrap_or(f64::NAN) < 1.0e-6);
        assert!(position.rms_residual_m.unwrap_or(f64::NAN) < 1.0e-9);
        assert!(position.gdop.unwrap_or(f64::NAN).is_finite());
    }

    #[test]
    fn positions_publish_coplanar_with_mirror_flag() {
        let microphones = vec![
            ("mic-0", [0.0, 0.0, 0.0]),
            ("mic-1", [2.0, 0.0, 0.0]),
            ("mic-2", [0.0, 2.0, 0.0]),
            ("mic-3", [2.0, 2.0, 0.0]),
        ];
        let truth = [1.0, 0.5, 1.5];
        let row = position_row(
            "left",
            microphones
                .iter()
                .map(|(id, position)| (*id, Some((distance_meter(*position, truth), 0.001))))
                .collect(),
        );
        let plan = position_plan(&microphones, None);
        let (sources, _) = solve_source_positions(&plan, &[row], 343.0, 48_000);
        let position = &sources[0];
        let fit = position.position_m.expect("coplanar ranges solve");
        assert!(position.mirror_ambiguous);
        assert_eq!(position.reason, None);
        assert!((fit[0] - truth[0]).abs() < 1.0e-6);
        assert!((fit[1] - truth[1]).abs() < 1.0e-6);
        assert!((fit[2].abs() - truth[2]).abs() < 1.0e-6);
    }

    #[test]
    fn positions_refuse_three_ranges_as_mirror() {
        let truth = [2.3, -1.1, 0.7];
        let (microphones, mut row) = tetra_row(truth, 0.001);
        row.cells[3].distance_m = None;
        row.cells[3].reason = Some("test_refused".into());
        let plan = position_plan(&microphones, None);
        let (sources, _) = solve_source_positions(&plan, &[row], 343.0, 48_000);
        let position = &sources[0];
        assert_eq!(position.position_m, None);
        assert!(position.mirror_ambiguous);
        assert_eq!(position.reason.as_deref(), Some("mirror_ambiguous"));
    }

    #[test]
    fn positions_refuse_two_ranges_as_insufficient() {
        let truth = [2.3, -1.1, 0.7];
        let (microphones, mut row) = tetra_row(truth, 0.001);
        for cell in row.cells.iter_mut().take(2) {
            cell.distance_m = None;
            cell.reason = Some("test_refused".into());
        }
        let plan = position_plan(&microphones, None);
        let (sources, _) = solve_source_positions(&plan, &[row], 343.0, 48_000);
        let position = &sources[0];
        assert_eq!(position.position_m, None);
        assert!(!position.mirror_ambiguous);
        assert_eq!(position.reason.as_deref(), Some("insufficient_ranges"));
    }

    #[test]
    fn positions_refuse_collinear_geometry() {
        let microphones = vec![
            ("mic-0", [0.0, 0.0, 0.0]),
            ("mic-1", [1.0, 0.0, 0.0]),
            ("mic-2", [2.0, 0.0, 0.0]),
            ("mic-3", [3.0, 0.0, 0.0]),
        ];
        let truth = [0.5, 1.0, 0.0];
        let row = position_row(
            "left",
            microphones
                .iter()
                .map(|(id, position)| (*id, Some((distance_meter(*position, truth), 0.001))))
                .collect(),
        );
        let plan = position_plan(&microphones, None);
        let (sources, _) = solve_source_positions(&plan, &[row], 343.0, 48_000);
        let position = &sources[0];
        assert_eq!(position.position_m, None);
        assert_eq!(position.reason.as_deref(), Some("degenerate_geometry"));
    }

    #[test]
    fn positions_refuse_depleted_fit_as_solve_failed() {
        let truth = [2.3, -1.1, 0.7];
        let (microphones, mut row) = tetra_row(truth, 0.001);
        let shifted = row.cells[3].distance_m.unwrap_or(0.0) + 0.05;
        row.cells[3].distance_m = Some(shifted);
        let plan = position_plan(&microphones, None);
        let (sources, issues) = solve_source_positions(&plan, &[row], 343.0, 48_000);
        let position = &sources[0];
        assert_eq!(position.position_m, None);
        assert_eq!(position.reason.as_deref(), Some("solve_failed"));
        assert!(issues.iter().any(|issue| issue.contains("solve_failed")));
    }

    #[test]
    fn positions_refuse_high_residual_with_evidence() {
        let truth = [2.3, -1.1, 0.7];
        let (microphones, _) = tetra_row(truth, 0.001);
        // Three tight ranges pin the fit; the loose fourth carries a large
        // self-consistent residual that trips Q7 without tripping exclusion.
        let row = position_row(
            "left",
            microphones
                .iter()
                .enumerate()
                .map(|(index, (id, position))| {
                    let distance = distance_meter(*position, truth);
                    if index == 3 {
                        (*id, Some((distance + 0.2, 0.1)))
                    } else {
                        (*id, Some((distance, 0.001)))
                    }
                })
                .collect(),
        );
        let plan = position_plan(&microphones, None);
        let (sources, issues) = solve_source_positions(&plan, &[row], 343.0, 48_000);
        let position = &sources[0];
        assert_eq!(position.position_m, None);
        assert_eq!(position.reason.as_deref(), Some("solve_residual_high"));
        assert!(position.rms_residual_m.unwrap_or(0.0) > 0.05);
        assert!(position.gdop.unwrap_or(f64::NAN).is_finite());
        assert!(
            issues
                .iter()
                .any(|issue| issue.contains("solve_residual_high"))
        );
    }

    #[test]
    fn readiness_flags_geometry_and_mode() {
        let mut plan: CaptureSessionPlan =
            serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json"))
                .unwrap();
        let lines = localization_readiness(&plan);
        assert!(lines[0].contains("2 microphones"));
        assert!(lines[0].contains("relative-only"));
        assert!(lines.iter().any(|line| line.contains("2/4 microphones")));
        let template = plan.microphones[0].clone();
        plan.microphones = [
            [0.0, 0.0, 1.1],
            [0.6, 0.0, 1.1],
            [0.3, 0.5, 1.1],
            [0.3, -0.4, 1.1],
        ]
        .iter()
        .enumerate()
        .map(|(channel, position)| {
            let mut microphone = template.clone();
            microphone.id = format!("mic-{channel}");
            microphone.input_channel = channel as u16;
            microphone.position_m = *position;
            microphone
        })
        .collect();
        plan.loopback = Some(crate::capture_session::CaptureLoopback {
            device: "iface".into(),
            input_channel: 7,
        });
        let lines = localization_readiness(&plan);
        assert!(lines[0].contains("loopback-referenced"));
        assert!(lines.iter().any(|line| line.contains("mirror-ambiguous")));
    }

    #[test]
    fn summary_lines_report_positions() {
        let report = LocalizationReport {
            mode: LocalizationMode::LoopbackReferenced,
            sound_speed_m_s: 343.0,
            sound_speed_uncertainty_m_s: 0.5,
            sound_speed_source: "plan".into(),
            pooled_skew_ppm: None,
            mic_bias_correction_m: 0.0,
            ranges: vec![
                SourceRanges {
                    source_id: "left".into(),
                    cells: Vec::new(),
                },
                SourceRanges {
                    source_id: "right".into(),
                    cells: Vec::new(),
                },
            ],
            sources: vec![
                SourcePosition {
                    source_id: "left".into(),
                    position_m: Some([2.3, -1.1, 0.7]),
                    sigma_m: Some([0.01, 0.01, 0.02]),
                    rms_residual_m: Some(0.003),
                    gdop: Some(2.1),
                    mirror_ambiguous: true,
                    declared_error_m: None,
                    reason: None,
                },
                SourcePosition {
                    source_id: "right".into(),
                    position_m: None,
                    sigma_m: None,
                    rms_residual_m: None,
                    gdop: None,
                    mirror_ambiguous: false,
                    declared_error_m: None,
                    reason: Some("solve_failed".into()),
                },
            ],
            issues: Vec::new(),
        };
        let lines = localization_summary_lines(&report);
        assert_eq!(lines.len(), 5);
        assert!(lines[2].contains("position: [2.300, -1.100, 0.700]"));
        assert!(lines[2].contains("rms 3.0 mm"));
        assert!(lines[2].contains("mirror-ambiguous"));
        assert!(lines[4].contains("refused (solve_failed)"));
    }
}
