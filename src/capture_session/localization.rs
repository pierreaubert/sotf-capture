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
use math_audio_dsp::capture_localization::{
    JointEstimate, JointPoint, MIN_JOINT_ANCHORS, MIN_RANGES_FOR_FIX, PositionEstimate,
    geometry_rank, joint_self_calibrate, loo_diagnostics, solve_position,
};
use math_audio_dsp::capture_tdoa::MIN_TDOA_CONFIDENCE_DB;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Write as _};
use std::path::{Path, PathBuf};

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
    /// Largest leave-one-out held-out residual in meters, when the
    /// geometry has spare microphones to re-solve without.
    #[serde(default)]
    pub loo_max_residual_m: Option<f64>,
    /// Microphone held out for the largest residual, when diagnosed.
    #[serde(default)]
    pub loo_worst_microphone: Option<String>,
    /// Largest leave-one-out solution shift in meters, when diagnosed.
    #[serde(default)]
    pub loo_max_shift_m: Option<f64>,
    /// Stable machine token explaining a missing position.
    #[serde(default)]
    pub reason: Option<String>,
}

/// One estimated microphone position from multilateration over a range
/// column (Phase 4 inverse mode) or the joint calibration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicPosition {
    /// Microphone label from the session declaration.
    pub microphone_id: String,
    /// Estimated microphone position in meters.
    #[serde(default)]
    pub position_m: Option<[f64; 3]>,
    /// Per-axis one-sigma uncertainty in meters.
    #[serde(default)]
    pub sigma_m: Option<[f64; 3]>,
    /// RMS range residual of the multilateration fit in meters.
    #[serde(default)]
    pub rms_residual_m: Option<f64>,
    /// Geometric dilution of precision of the source geometry.
    #[serde(default)]
    pub gdop: Option<f64>,
    /// True when coplanar sources leave a mirror solution.
    pub mirror_ambiguous: bool,
    /// Distance to the surveyed microphone position; always `None`
    /// (surveyed microphones are anchors, never estimates).
    #[serde(default)]
    pub declared_error_m: Option<f64>,
    /// Largest leave-one-out held-out residual in meters, when the
    /// geometry has spare sources to re-solve without.
    #[serde(default)]
    pub loo_max_residual_m: Option<f64>,
    /// Source held out for the largest residual, when diagnosed.
    #[serde(default)]
    pub loo_worst_source: Option<String>,
    /// Largest leave-one-out solution shift in meters, when diagnosed.
    #[serde(default)]
    pub loo_max_shift_m: Option<f64>,
    /// Stable machine token explaining a missing position.
    #[serde(default)]
    pub reason: Option<String>,
}

/// One joint self-calibration summary (Phase 4): how many points on
/// each side the joint solve estimated, from how many anchors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointCalibration {
    /// Unsurveyed sources the joint solve estimated.
    pub sources_solved: usize,
    /// Unsurveyed microphones the joint solve estimated.
    pub microphones_solved: usize,
    /// Surveyed anchors (sources plus microphones) fixing the frame.
    pub anchors: usize,
    /// RMS residual over all jointly used cells in meters.
    pub rms_residual_m: f64,
    /// True unless the anchors span rank 3: a 3-anchor (or any
    /// coplanar-anchor) frame carries a global mirror across the
    /// anchor plane with identical residuals.
    pub mirror_ambiguous: bool,
    /// True when the surveyed coverage was too thin for consensus
    /// cascade seeding: a gross outlier may have absorbed into the
    /// frame with clean residuals, so verify the geometry when
    /// outliers are possible.
    pub outlier_absorb_risk: bool,
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
    /// One entry per plan microphone, in plan order (Phase 4
    /// inverse/joint microphone positions).
    #[serde(default)]
    pub microphones: Vec<MicPosition>,
    /// Joint self-calibration summary when both sides had unsurveyed
    /// points and the joint solve ran (§6.4); `None` for direct-only
    /// sessions and refused or not-attempted joint solves.
    #[serde(default)]
    pub joint: Option<JointCalibration>,
    /// Session-level evidence notes and degradation warnings.
    #[serde(default)]
    pub issues: Vec<String>,
}

/// Residual matched-microphone bias after bench correction (§6.3 budget).
///
/// Applied to sigma only when a measured bench bias is in force;
/// uncalibrated sessions budget nothing and disclose the zero correction
/// through `LocalizationReport.mic_bias_correction_m` instead.
const BENCH_BIAS_RESIDUAL_M: f64 = 0.002;

/// Expected bench calibration schema (`docs/localization-bench-calibration.md` §3).
const BENCH_CALIBRATION_SCHEMA: &str = "sotf-capture/localization-calibration";

/// Supported bench calibration schema version.
const BENCH_CALIBRATION_VERSION: u32 = 1;

/// One-time bench calibration for an interface + microphone model (§5.4).
///
/// Loaded from the plan's `localization_calibration_file`; any load or
/// validation failure falls back to uncalibrated (zero bias, no latency
/// bound) with a session issue, never an aborted run.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchCalibration {
    /// Interface identity from the file, informational only.
    pub interface: String,
    /// Microphone model from the file, informational only.
    pub microphone_model: String,
    /// Bench loopback lag in samples by sample rate in Hz.
    pub loopback_lag_samples: BTreeMap<u32, f64>,
    /// Bench mic bias in meters; meaningful only when `mic_bias_measured`.
    pub mic_bias_correction_m: f64,
    /// Whether the bias was actually measured (`mic_bias_measured`).
    /// An unmeasured value applies nothing even when present.
    pub mic_bias_measured: bool,
}

/// Raw bench file shape; validated field by field in
/// [`load_bench_calibration`] so failures name the offending entry.
#[derive(Debug, Deserialize)]
struct BenchCalibrationFile {
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    schema_version: Option<u32>,
    #[serde(default)]
    interface: Option<String>,
    #[serde(default)]
    microphone_model: Option<String>,
    #[serde(default)]
    latency_samples_by_rate_hz: BTreeMap<String, f64>,
    #[serde(default)]
    mic_bias_correction_m: Option<f64>,
    #[serde(default)]
    mic_bias_measured: Option<bool>,
}

/// Load and validate a bench calibration file (§5.4 format).
///
/// Unknown extra fields are ignored for forward compatibility;
/// unknown schema versions are refused, never guessed.
///
/// # Errors
///
/// Rejects unreadable files, malformed JSON, wrong schemas and
/// versions, unparsable rate keys, non-finite or negative lags, a
/// non-finite bias, a measured flag without a value, and files
/// providing neither a lag map nor a measured bias.
pub fn load_bench_calibration(path: &Path) -> Result<BenchCalibration, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let file: BenchCalibrationFile = serde_json::from_str(&json)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
    if file.schema.as_deref() != Some(BENCH_CALIBRATION_SCHEMA) {
        return Err(format!("schema must be {BENCH_CALIBRATION_SCHEMA}"));
    }
    if file.schema_version != Some(BENCH_CALIBRATION_VERSION) {
        return Err(format!(
            "unsupported schema version {:?}",
            file.schema_version
        ));
    }
    let mut loopback_lag_samples = BTreeMap::new();
    for (rate, lag) in &file.latency_samples_by_rate_hz {
        let rate_hz: u32 = rate
            .parse()
            .map_err(|_| format!("rate key {rate:?} is not a sample rate"))?;
        if !lag.is_finite() || *lag < 0.0 {
            return Err(format!(
                "loopback lag for {rate_hz} Hz must be finite and nonnegative"
            ));
        }
        loopback_lag_samples.insert(rate_hz, *lag);
    }
    let measured = file.mic_bias_measured.unwrap_or(false);
    if measured && file.mic_bias_correction_m.is_none() {
        return Err("mic_bias_measured is true but mic_bias_correction_m is absent".into());
    }
    if let Some(bias) = file.mic_bias_correction_m {
        if !bias.is_finite() {
            return Err("mic_bias_correction_m must be finite".into());
        }
    }
    if loopback_lag_samples.is_empty() && !measured {
        return Err("bench file provides neither loopback lag nor measured mic bias".into());
    }
    Ok(BenchCalibration {
        interface: file.interface.unwrap_or_default(),
        microphone_model: file.microphone_model.unwrap_or_default(),
        loopback_lag_samples,
        mic_bias_correction_m: file.mic_bias_correction_m.unwrap_or(0.0),
        mic_bias_measured: measured,
    })
}

/// Resolve a bench calibration path for processing.
///
/// Absolute paths pass through; relative paths resolve against the raw
/// session root, where the operator stages the per-interface file
/// alongside the takes. `plan-validate` checks the same path against
/// the plan directory instead, so prefer absolute paths for bench
/// files and treat the validate check as readability guidance.
fn resolve_bench_path(raw_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        raw_root.join(path)
    }
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
/// Low-band edges for subwoofer arrivals in Hz (§4.3 caveat).
const SUB_BAND_LO_HZ: f64 = 20.0;
/// See [`SUB_BAND_LO_HZ`].
const SUB_BAND_HI_HZ: f64 = 300.0;
/// Minimum usable low-band width in Hz: a third of the 280 Hz band,
/// mirroring the design-band ratio.
const SUB_MIN_BANDWIDTH_HZ: f64 = 100.0;
/// Low-band bandpass length in taps: the 127-tap design cannot resolve
/// a 300 Hz cutoff at 48 kHz (transition wider than the passband), so
/// the sub band gets a longer filter of the identical windowed-sinc
/// design (still linear-phase, so group delay still cancels).
const SUB_FIR_TAPS: usize = 1025;
/// Detector cutoff in Hz: recorded sweep energy above this counts as
/// high-frequency content.
const SUB_HF_CUTOFF_HZ: f64 = 300.0;
/// Detector threshold: an HF energy fraction below this routes the
/// take to the low-band estimator. Errs toward the standard estimator —
/// weak-HF takes refuse there rather than misroute here.
const SUB_HF_FRACTION: f64 = 0.05;

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
/// The tap count coerces to an odd number of at least three, keeping
/// the phase exactly linear on every input.
#[expect(
    clippy::cast_precision_loss,
    reason = "Tap counts convert to f64 exactly; far below precision limits."
)]
fn arrival_bandpass(rate: f64, band_lo_hz: f64, band_hi_hz: f64, taps: usize) -> Vec<f64> {
    // Integer center: with odd length the sinc and the Hann window share
    // tap (N-1)/2, which is what makes the phase exactly linear.
    let taps = taps.max(3) | 1;
    let middle = (taps - 1) as f64 / 2.0;
    (0..taps)
        .map(|tap| {
            let offset = tap as f64 - middle;
            let sinc = if offset.abs() < 1e-9 {
                2.0 * (band_hi_hz - band_lo_hz) / rate
            } else {
                let phase = std::f64::consts::PI * offset / rate;
                ((2.0 * band_hi_hz * phase).sin() - (2.0 * band_lo_hz * phase).sin())
                    / (std::f64::consts::PI * offset)
            };
            let window =
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * tap as f64 / (taps - 1) as f64).cos();
            sinc * window
        })
        .collect()
}

/// Usable low-band for one sweep, intersected with the sub band.
///
/// Returns `None` when the sweep leaves less than
/// [`SUB_MIN_BANDWIDTH_HZ`] inside 20–300 Hz: the Cramér–Rao bound
/// degrades with bandwidth, and below ~100 Hz the arrival carries no
/// usable timing.
fn sub_band(sweep_lo_hz: f64, sweep_hi_hz: f64) -> Option<(f64, f64)> {
    if !sweep_lo_hz.is_finite() || !sweep_hi_hz.is_finite() || sweep_hi_hz <= sweep_lo_hz {
        return None;
    }
    let lo = sweep_lo_hz.max(SUB_BAND_LO_HZ);
    let hi = sweep_hi_hz.min(SUB_BAND_HI_HZ);
    if hi - lo >= SUB_MIN_BANDWIDTH_HZ {
        Some((lo, hi))
    } else {
        None
    }
}

/// Remove the least-squares line from a signal.
///
/// Kills DC and slow rumble drift before low-band correlation without
/// touching phase: a fitted-line subtraction is zero-phase, so the
/// correlation peak does not move. Empty input maps to empty output.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Sample indices convert to f64 coordinates and detrended samples return to input precision; bounded recordings stay far below precision limits."
)]
fn detrend_linear(signal: &[f32]) -> Vec<f32> {
    let count = signal.len();
    if count == 0 {
        return Vec::new();
    }
    let n = count as f64;
    let mean_position = (n - 1.0) / 2.0;
    let mean_value = signal.iter().map(|sample| f64::from(*sample)).sum::<f64>() / n;
    let covariance = signal
        .iter()
        .enumerate()
        .map(|(index, sample)| (index as f64 - mean_position) * (f64::from(*sample) - mean_value))
        .sum::<f64>();
    let variance = (0..count)
        .map(|index| (index as f64 - mean_position).powi(2))
        .sum::<f64>();
    let (slope, intercept) = if variance.abs() > 1e-9 {
        let slope = covariance / variance;
        (slope, mean_value - slope * mean_position)
    } else {
        (0.0, mean_value)
    };
    signal
        .iter()
        .enumerate()
        .map(|(index, sample)| (f64::from(*sample) - (slope * index as f64 + intercept)) as f32)
        .collect()
}

/// Fraction of signal energy above a cutoff in Hz (§4.3 detector).
///
/// Hann-windowed DFT power ratio (the window's coherent gain cancels
/// in the ratio; Parseval needs no scaling for a relative measure).
/// Measures the RECORDED take, never the sweep config, so a
/// subwoofer that cannot reproduce the sweep's top end routes
/// low-band while a full-range take of the same sweep does not.
/// Returns `None` on empty input, silence, or degenerate parameters,
/// which routes the standard estimator (the safe default).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "Bin counts convert to f64 coordinates; the cutoff bin is non-negative by the validated cutoff and rate, and bounded recordings stay far below precision limits."
)]
fn hf_energy_fraction(signal: &[f32], cutoff_hz: f64, rate: f64) -> Option<f64> {
    use rustfft::{FftPlanner, num_complex::Complex};
    if signal.len() < 64
        || !cutoff_hz.is_finite()
        || cutoff_hz <= 0.0
        || !rate.is_finite()
        || rate <= 0.0
        || cutoff_hz >= rate / 2.0
    {
        return None;
    }
    let fft_len = signal.len().next_power_of_two();
    let mut spectrum: Vec<Complex<f32>> = signal
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let window =
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * index as f64 / signal.len() as f64).cos();
            Complex::new(*sample * window as f32, 0.0)
        })
        .collect();
    spectrum.resize(fft_len, Complex::new(0.0, 0.0));
    FftPlanner::<f32>::new()
        .plan_fft_forward(fft_len)
        .process(&mut spectrum);
    let total: f64 = spectrum.iter().map(|bin| f64::from(bin.norm_sqr())).sum();
    if total <= 0.0 {
        return None;
    }
    let cutoff_bin = ((cutoff_hz * fft_len as f64 / rate).ceil() as usize).min(fft_len / 2);
    let high: f64 = spectrum
        .iter()
        .enumerate()
        .filter(|(bin, _)| (*bin).min(fft_len - *bin) >= cutoff_bin)
        .map(|(_, bin)| f64::from(bin.norm_sqr()))
        .sum();
    Some(high / total)
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
/// within one sample (Q3). Takes whose recorded sweep carries negligible
/// energy above 300 Hz route to the 20–300 Hz low band instead
/// (detrended first; uncertainty widens through the Cramér–Rao
/// bandwidth term). Refusals carry stable reason tokens: a missing
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
    // Sub-band routing (§4.3 caveat): a recorded sweep with negligible
    // energy above 300 Hz goes low-band. Detection runs on the sweep
    // core (chirps excluded by layout) and measures the take, never
    // the sweep config; anything unmeasurable keeps the standard
    // estimator, which is the safe default.
    let core_start = predicted_start.max(0.0) as usize;
    let core_stop = (predicted_start + sweep.len() as f64)
        .min(recorded.len() as f64)
        .max(0.0) as usize;
    let core = recorded
        .get(core_start..core_stop)
        .filter(|core| core.len() >= 64);
    let sub = core
        .and_then(|core| hf_energy_fraction(core, SUB_HF_CUTOFF_HZ, f64::from(rate)))
        .is_some_and(|fraction| fraction < SUB_HF_FRACTION);
    let (band_lo, band_hi, tap_count) = if sub {
        let Some(band) = sub_band(sweep_lo_hz, sweep_hi_hz) else {
            return Err("band_unsupported");
        };
        (band.0, band.1, SUB_FIR_TAPS)
    } else {
        let Some(band) = arrival_band(sweep_lo_hz, sweep_hi_hz) else {
            return Err("band_unsupported");
        };
        (band.0, band.1, ARRIVAL_FIR_TAPS)
    };
    let taps = arrival_bandpass(f64::from(rate), band_lo, band_hi, tap_count);
    // Detrend before the low band: DC and rumble drift would otherwise
    // bias the low-frequency correlation pedestal.
    let (reference, windowed) = if sub {
        (
            apply_arrival_bandpass(&detrend_linear(sweep), &taps),
            apply_arrival_bandpass(&detrend_linear(window), &taps),
        )
    } else {
        (
            apply_arrival_bandpass(sweep, &taps),
            apply_arrival_bandpass(window, &taps),
        )
    };
    let correlation = cross_correlate_envelope(&reference, &windowed, rate)
        .map_err(|_| "arrival_low_confidence")?;
    let confidence_db = f64::from(correlation.peak_to_sidelobe_db);
    if confidence_db.is_nan() || confidence_db < MIN_TDOA_CONFIDENCE_DB {
        return Err("arrival_low_confidence");
    }
    let strongest = start as f64 + correlation.peak_sample_refined;
    // A low-frequency hump carries ripple modulation across a wide
    // pedestal: a first-crossing direct pick strands on ripples while
    // the hump peak is unambiguous, so the sub-band arrival IS the
    // peak. Early reflections inside the LF resolution merge into the
    // hump and are budgeted through the CRLB sigma, not gated here —
    // demanding single-sample direct/strongest agreement at LF refuses
    // every take, valid or not.
    let direct_at = if sub {
        strongest
    } else {
        // Two rise times complete a full-band hump without reaching
        // late reflections; the standard window still binds, so
        // full-band behavior is unchanged.
        let rise = (2.0 * f64::from(rate) / (band_hi - band_lo)).round();
        let lookahead =
            crate::signal_recorder::standard_direct_lookahead_samples(rate).max(rise as usize);
        let direct = crate::signal_recorder::pick_direct_arrival_with_lookahead(
            &correlation.envelope,
            rate,
            windowed.len(),
            lookahead,
        )
        .ok_or("arrival_low_confidence")?;
        let direct_at = start as f64 + direct.peak_sample_refined;
        if (direct_at - strongest).abs() > ARRIVAL_AGREEMENT_SAMPLES {
            return Err("arrival_ambiguous");
        }
        direct_at
    };
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
/// Subtracts the declared source electronic delay when present and the
/// bench mic-bias correction (§7.2 step 6). A non-finite result refuses
/// the cell downstream; the bias itself is validated at load time.
fn propagation_to_meters(
    prop_samples: f64,
    rate: f64,
    sound_speed: f64,
    source: &CaptureSource,
    mic_bias_m: f64,
) -> Option<f64> {
    if !prop_samples.is_finite() || !rate.is_finite() || rate <= 0.0 {
        return None;
    }
    let mut prop = prop_samples;
    if let Some(delay_ms) = source.source_delay_calibration_ms {
        prop -= delay_ms / 1000.0 * rate;
    }
    let distance = prop / rate * sound_speed - mic_bias_m;
    distance.is_finite().then_some(distance)
}

/// Combine timing uncertainty with sound-speed uncertainty in meters.
///
/// Worst-cased sum, mirroring the clock stage: arrival/clock timing
/// converted at the sound speed plus the fractional sound-speed error
/// times the distance plus the bench bias residual (zero when
/// uncalibrated). Returns `None` when timing is unknown.
fn range_sigma_m(
    sigma_time_samples: f64,
    distance: f64,
    rate: f64,
    sound_speed: f64,
    sound_speed_uncertainty: f64,
    mic_bias_residual_m: f64,
) -> Option<f64> {
    if !sigma_time_samples.is_finite() || sigma_time_samples < 0.0 {
        return None;
    }
    let sigma = sound_speed * sigma_time_samples / rate
        + distance * sound_speed_uncertainty / sound_speed
        + mic_bias_residual_m;
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
        let Some(mic_position) = mic.position_m else {
            continue;
        };
        let surveyed = mic_position
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
fn refused_cell(microphone_id: &str, reason: &str) -> RangeCell {
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
    mic_bias_m: f64,
    mic_bias_residual_m: f64,
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
        let Some(distance) = propagation_to_meters(prop, rate, sound_speed, source, mic_bias_m)
        else {
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
                mic_bias_residual_m,
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
#[expect(
    clippy::too_many_arguments,
    reason = "Mode conversion threads the decided session evidence explicitly; bundling would hide the per-cell inputs."
)]
fn emitter_cells(
    report: &ClockProcessedManifest,
    group: &SelectedGroup,
    fits: &[Option<CaptureClockFit>],
    arrivals: &BTreeMap<usize, Result<SweepArrival, &'static str>>,
    sound_speed: f64,
    sound_speed_uncertainty: f64,
    max_range: f64,
    mic_bias_m: f64,
    mic_bias_residual_m: f64,
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
        let Some(distance) = propagation_to_meters(prop, rate, sound_speed, source, mic_bias_m)
        else {
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
                mic_bias_residual_m,
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

/// Build one per-repeat source row for the decided session mode.
///
/// A repeat whose loopback is unusable yields an all-refused row with
/// `loopback_unavailable`; the repeat combination below treats those
/// cells as non-voters instead of failing the source.
#[expect(
    clippy::too_many_arguments,
    reason = "Mode dispatch threads the decided session evidence explicitly; bundling would hide the per-cell inputs."
)]
fn build_group_row(
    report: &ClockProcessedManifest,
    group: &SelectedGroup,
    mode: LocalizationMode,
    loopbacks: &BTreeMap<(usize, u32), Result<GroupLoopback, String>>,
    fits: &[Option<CaptureClockFit>],
    arrivals: &BTreeMap<usize, Result<SweepArrival, &'static str>>,
    sound_speed: f64,
    sound_speed_uncertainty: f64,
    max_range: f64,
    skew_ratio: f64,
    mic_bias_m: f64,
    mic_bias_residual_m: f64,
) -> SourceRanges {
    match mode {
        LocalizationMode::LoopbackReferenced => {
            let Some(Ok(loopback)) = loopbacks.get(&(group.source_index, group.repeat_index))
            else {
                let source = &report.plan.sources[group.source_index];
                return SourceRanges {
                    source_id: source.id.clone(),
                    cells: report
                        .plan
                        .microphones
                        .iter()
                        .map(|mic| refused_cell(&mic.id, "loopback_unavailable"))
                        .collect(),
                };
            };
            loopback_cells(
                report,
                group,
                loopback,
                arrivals,
                sound_speed,
                sound_speed_uncertainty,
                max_range,
                skew_ratio,
                mic_bias_m,
                mic_bias_residual_m,
            )
        }
        LocalizationMode::EmitterReferenced => emitter_cells(
            report,
            group,
            fits,
            arrivals,
            sound_speed,
            sound_speed_uncertainty,
            max_range,
            mic_bias_m,
            mic_bias_residual_m,
        ),
        LocalizationMode::RelativeOnly => relative_cells(report, group, arrivals, skew_ratio),
    }
}

/// Combine per-repeat rows for one source into a single row (§7.2 step 5).
///
/// Zero repeats (a source with no takes at all) refuse every cell with
/// `take_missing`; one repeat passes its row through unchanged so
/// single-repeat sessions keep bit-identical output; several repeats
/// combine per microphone in [`combine_mic_cells`].
fn combine_repeat_rows(
    plan: &CaptureSessionPlan,
    source: &CaptureSource,
    rows: &[SourceRanges],
) -> SourceRanges {
    match rows {
        [] => SourceRanges {
            source_id: source.id.clone(),
            cells: plan
                .microphones
                .iter()
                .map(|mic| refused_cell(&mic.id, "take_missing"))
                .collect(),
        },
        [only] => only.clone(),
        _ => SourceRanges {
            source_id: source.id.clone(),
            cells: plan
                .microphones
                .iter()
                .enumerate()
                .map(|(mic_index, mic)| {
                    let per_repeat: Vec<&RangeCell> = rows
                        .iter()
                        .filter_map(|row| row.cells.get(mic_index))
                        .filter(|cell| cell.microphone_id == mic.id)
                        .collect();
                    combine_mic_cells(&mic.id, &per_repeat)
                })
                .collect(),
        },
    }
}

/// Combine one microphone's per-repeat cells.
///
/// The median arrival (equivalently, the median distance — medians
/// commute with the affine delay-to-distance conversion) rejects a wild
/// repeat; the inter-repeat spread widens sigma in quadrature with the
/// median estimator sigma, so disagreement visibly downweights the cell
/// instead of silently averaging away. Evidence takes the median arrival
/// and reference, the minimum confidence, and the most common band.
/// A microphone with no usable repeat keeps the most common refusal
/// token. Deterministic throughout, total on all inputs.
fn combine_mic_cells(microphone_id: &str, cells: &[&RangeCell]) -> RangeCell {
    let mut distances: Vec<f64> = cells.iter().filter_map(|cell| cell.distance_m).collect();
    distances.sort_by(f64::total_cmp);
    let mut sigmas: Vec<f64> = cells.iter().filter_map(|cell| cell.sigma_m).collect();
    sigmas.sort_by(f64::total_cmp);
    let mut arrivals: Vec<f64> = cells
        .iter()
        .filter_map(|cell| cell.arrival_samples)
        .collect();
    arrivals.sort_by(f64::total_cmp);
    let mut references: Vec<f64> = cells
        .iter()
        .filter_map(|cell| cell.reference_arrival_samples)
        .collect();
    references.sort_by(f64::total_cmp);
    let mut tdoas: Vec<f64> = cells
        .iter()
        .filter_map(|cell| cell.tdoa_samples_vs_first)
        .collect();
    tdoas.sort_by(f64::total_cmp);
    let mut tdoa_sigmas: Vec<f64> = cells
        .iter()
        .filter_map(|cell| cell.tdoa_sigma_samples)
        .collect();
    tdoa_sigmas.sort_by(f64::total_cmp);
    if distances.is_empty() && tdoas.is_empty() {
        let token = most_common(cells.iter().filter_map(|cell| cell.reason.as_deref()))
            .unwrap_or("take_missing");
        return refused_cell(microphone_id, token);
    }
    let distance_m = median_sorted(&distances);
    let sigma_m = widened_sigma(median_sorted(&sigmas), half_spread(&distances));
    let tdoa_samples_vs_first = median_sorted(&tdoas);
    let tdoa_sigma_samples = widened_sigma(median_sorted(&tdoa_sigmas), half_spread(&tdoas));
    let confidence_db = cells
        .iter()
        .filter_map(|cell| cell.confidence_db)
        .reduce(f64::min);
    RangeCell {
        microphone_id: microphone_id.into(),
        distance_m,
        sigma_m,
        arrival_samples: median_sorted(&arrivals),
        reference_arrival_samples: median_sorted(&references),
        tdoa_samples_vs_first,
        tdoa_sigma_samples,
        band: most_common(cells.iter().filter_map(|cell| cell.band.clone())),
        confidence_db,
        reason: most_common(cells.iter().filter_map(|cell| cell.reason.as_deref()))
            .map(str::to_string),
    }
}

/// Half the max-min spread of an ascending slice; 0.0 for fewer than two
/// values (a single repeat measures no spread).
fn half_spread(sorted: &[f64]) -> f64 {
    if sorted.len() < 2 {
        return 0.0;
    }
    (sorted[sorted.len() - 1] - sorted[0]) / 2.0
}

/// Median estimator sigma widened in quadrature with the measured
/// inter-repeat spread. Unknown estimator sigma with a measured spread
/// falls back to the spread itself; nothing known stays unknown.
fn widened_sigma(median_sigma: Option<f64>, spread_half: f64) -> Option<f64> {
    match (median_sigma, spread_half) {
        (Some(sigma), spread) => Some((sigma.powi(2) + spread.powi(2)).sqrt()),
        (None, spread) if spread > 0.0 => Some(spread),
        (None, _) => None,
    }
}

/// Most common item; ties break toward the smallest (deterministic).
fn most_common<T: Ord + Clone>(items: impl Iterator<Item = T>) -> Option<T> {
    let mut counts: BTreeMap<T, usize> = BTreeMap::new();
    for item in items {
        *counts.entry(item).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by(|left, right| left.1.cmp(&right.1).then(right.0.cmp(&left.0)))
        .map(|(item, _)| item)
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
        microphones: Vec::new(),
        joint: None,
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
    let all_groups = capture_groups(report);
    // Arrivals cover every repeat: non-selected repeats still vote in
    // the per-source median (§7.2 step 5), so their clock outliers are
    // reported rather than silently ignored.
    let mut arrivals: BTreeMap<usize, Result<SweepArrival, &'static str>> = BTreeMap::new();
    for group in &all_groups {
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
    // Bench calibration (§5.4): a measured bias shifts every absolute
    // range and widens sigma by the residual budget; any load failure
    // falls back to uncalibrated with an issue, never an aborted run.
    // Relative paths resolve against the raw session root here (see
    // `resolve_bench_path`); `plan-validate` checks the plan directory.
    let bench = match report.plan.localization_calibration_file.as_ref() {
        Some(path) => match load_bench_calibration(&resolve_bench_path(raw_root, path)) {
            Ok(calibration) => Some(calibration),
            Err(error) => {
                issues.push(format!(
                    "localization bench calibration unusable ({error}); continuing uncalibrated"
                ));
                None
            }
        },
        None => None,
    };
    let (mic_bias_m, mic_bias_residual_m) = match &bench {
        Some(calibration) if calibration.mic_bias_measured => {
            (calibration.mic_bias_correction_m, BENCH_BIAS_RESIDUAL_M)
        }
        _ => (0.0, 0.0),
    };
    // One row per plan source: every repeat votes, combined by median
    // with the inter-repeat spread widening sigma (§7.2 step 5). Mode
    // stays decided from the representative groups above.
    let mut ranges = Vec::with_capacity(report.plan.sources.len());
    for (source_index, source) in report.plan.sources.iter().enumerate() {
        let repeats: Vec<SourceRanges> = all_groups
            .iter()
            .filter(|group| group.source_index == source_index)
            .map(|group| {
                build_group_row(
                    report,
                    group,
                    mode,
                    &loopbacks,
                    fits,
                    &arrivals,
                    sound_speed,
                    sound_speed_uncertainty,
                    max_range,
                    skew_ratio,
                    mic_bias_m,
                    mic_bias_residual_m,
                )
            })
            .collect();
        ranges.push(combine_repeat_rows(&report.plan, source, &repeats));
    }
    // Joint self-calibration runs only when both sides hold unsurveyed
    // points; one-sided sessions keep the direct solves, and Q10
    // counting failures fall back to them with a session note (Q10b
    // convergence failures refuse every unsurveyed point instead).
    let joint_attempt = (report
        .plan
        .sources
        .iter()
        .any(|source| source.position_m.is_none())
        && report
            .plan
            .microphones
            .iter()
            .any(|microphone| microphone.position_m.is_none()))
    .then(|| try_joint_positions(&report.plan, &ranges, sound_speed, rate));
    let (sources, microphones, joint) = match joint_attempt {
        Some(Ok(mut attempt)) => {
            issues.append(&mut attempt.notes);
            (attempt.sources, attempt.microphones, attempt.calibration)
        }
        attempt => {
            if let Some(Err(mut q10)) = attempt {
                issues.append(&mut q10);
            }
            let (sources, mut position_issues) =
                solve_source_positions(&report.plan, &ranges, sound_speed, rate);
            issues.append(&mut position_issues);
            let (microphones, mut mic_issues) =
                solve_microphone_positions(&report.plan, &ranges, sound_speed, rate);
            issues.append(&mut mic_issues);
            (sources, microphones, None)
        }
    };
    LocalizationReport {
        mode,
        sound_speed_m_s: sound_speed,
        sound_speed_uncertainty_m_s: sound_speed_uncertainty,
        sound_speed_source: sound_source.into(),
        pooled_skew_ppm: pooled,
        mic_bias_correction_m: mic_bias_m,
        ranges,
        sources,
        microphones,
        joint,
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

/// Solve one 3D position per microphone column, in plan order.
///
/// Inverse of [`solve_source_positions`]: columns align with plan
/// microphones by id, and each column solves against surveyed plan
/// sources. Surveyed microphones are anchors, never estimates.
fn solve_microphone_positions(
    plan: &CaptureSessionPlan,
    ranges: &[SourceRanges],
    sound_speed_m_s: f64,
    sample_rate_hz: u32,
) -> (Vec<MicPosition>, Vec<String>) {
    let sample_m = sound_speed_m_s / f64::from(sample_rate_hz.max(1));
    let mut microphones = Vec::with_capacity(plan.microphones.len());
    let mut issues = Vec::new();
    for (index, microphone) in plan.microphones.iter().enumerate() {
        let (position, mut notes) = solve_column_position(
            &microphone.id,
            index,
            ranges,
            &plan.sources,
            microphone.position_m,
            sample_m,
        );
        microphones.push(position);
        issues.append(&mut notes);
    }
    (microphones, issues)
}

/// Dense joint-solve inputs assembled from plan survey + range matrix.
struct JointInputs {
    source_seeds: Vec<Option<[f64; 3]>>,
    mic_seeds: Vec<Option<[f64; 3]>>,
    ranges_sm: Vec<Option<(f64, f64)>>,
    usable: Vec<bool>,
}

/// A joint attempt's published outputs.
///
/// `calibration` is `None` when the attempt itself failed past counting
/// (Q10b): every unsurveyed point is refused with `solve_failed` while
/// surveyed audits still publish.
struct JointAttempt {
    sources: Vec<SourcePosition>,
    microphones: Vec<MicPosition>,
    notes: Vec<String>,
    calibration: Option<JointCalibration>,
}

/// One range cell as a joint-solve link, or `None` when refused.
///
/// The validity rules mirror the solver's exactly (see
/// [`gather_usable_cells`]): finite non-negative distance with finite
/// positive sigma.
fn joint_usable_link(cell: &RangeCell) -> Option<(f64, f64)> {
    if let (Some(distance), Some(sigma)) = (cell.distance_m, cell.sigma_m)
        && distance.is_finite()
        && distance >= 0.0
        && sigma.is_finite()
        && sigma > 0.0
    {
        Some((distance, sigma))
    } else {
        None
    }
}

/// Assemble dense joint-solve inputs from the plan and range matrix.
///
/// Rows pair with plan sources by order and by id (a misaligned row
/// counts as missing, as in [`gather_usable_column`]); cells pair
/// with plan microphones the same way ([`gather_usable_cells`]).
fn build_joint_inputs(plan: &CaptureSessionPlan, ranges: &[SourceRanges]) -> JointInputs {
    let microphones = plan.microphones.len();
    let mut ranges_sm = Vec::with_capacity(plan.sources.len() * microphones);
    let mut usable = Vec::with_capacity(plan.sources.len() * microphones);
    for (index, source) in plan.sources.iter().enumerate() {
        let row = ranges.get(index).filter(|row| row.source_id == source.id);
        for (mic_index, microphone) in plan.microphones.iter().enumerate() {
            let link = row
                .and_then(|row| row.cells.get(mic_index))
                .filter(|cell| cell.microphone_id == microphone.id)
                .and_then(joint_usable_link);
            usable.push(link.is_some());
            ranges_sm.push(link);
        }
    }
    JointInputs {
        source_seeds: plan
            .sources
            .iter()
            .map(|source| source.position_m)
            .collect(),
        mic_seeds: plan
            .microphones
            .iter()
            .map(|microphone| microphone.position_m)
            .collect(),
        ranges_sm,
        usable,
    }
}

/// Q10 counting gate over assembled joint inputs (§9).
///
/// Mirrors the solver's refusal predicates exactly (same values, same
/// anchor order: sources then microphones), so a pass here means a
/// solver `None` is truly convergence (Q10b), never counting. Each
/// failing rule yields one session note; empty means attempt.
fn joint_counting_notes(plan: &CaptureSessionPlan, inputs: &JointInputs) -> Vec<String> {
    let mut notes = Vec::new();
    for (index, source) in plan.sources.iter().enumerate() {
        let surveyed = inputs.source_seeds.get(index).copied().flatten();
        if surveyed.is_some_and(|position| !position.iter().all(|c| c.is_finite())) {
            notes.push(format!(
                "joint self-calibration not attempted: surveyed source {} position is not finite; direct solves stand",
                source.id
            ));
        }
    }
    for (index, microphone) in plan.microphones.iter().enumerate() {
        let surveyed = inputs.mic_seeds.get(index).copied().flatten();
        if surveyed.is_some_and(|position| !position.iter().all(|c| c.is_finite())) {
            notes.push(format!(
                "joint self-calibration not attempted: surveyed microphone {} position is not finite; direct solves stand",
                microphone.id
            ));
        }
    }
    if !notes.is_empty() {
        return notes;
    }
    let anchors: Vec<[f64; 3]> = inputs
        .source_seeds
        .iter()
        .chain(inputs.mic_seeds.iter())
        .filter_map(|seed| *seed)
        .collect();
    if anchors.len() < MIN_JOINT_ANCHORS {
        notes.push(format!(
            "joint self-calibration not attempted: {} surveyed anchors (< {MIN_JOINT_ANCHORS} required); direct solves stand",
            anchors.len()
        ));
        return notes;
    }
    if geometry_rank(&anchors).collinear {
        notes.push(
            "joint self-calibration not attempted: surveyed anchors are collinear; direct solves stand"
                .into(),
        );
        return notes;
    }
    let unknowns = inputs
        .source_seeds
        .iter()
        .chain(inputs.mic_seeds.iter())
        .filter(|seed| seed.is_none())
        .count();
    let usable_count = inputs.usable.iter().filter(|used| **used).count();
    if usable_count < 3 * unknowns {
        notes.push(format!(
            "joint self-calibration not attempted: {usable_count} usable cells below the {} minimum (3 per unknown); direct solves stand",
            3 * unknowns
        ));
    }
    let mics = inputs.mic_seeds.len();
    for (index, source) in plan.sources.iter().enumerate() {
        if inputs.source_seeds.get(index).copied().flatten().is_some() {
            continue;
        }
        let links = (0..mics)
            .filter(|mic| {
                inputs
                    .usable
                    .get(index * mics + mic)
                    .copied()
                    .unwrap_or(false)
            })
            .count();
        if links < MIN_RANGES_FOR_FIX {
            notes.push(format!(
                "joint self-calibration not attempted: source {} has {links} usable links (< {MIN_RANGES_FOR_FIX} required); direct solves stand",
                source.id
            ));
        }
    }
    for (index, microphone) in plan.microphones.iter().enumerate() {
        if inputs.mic_seeds.get(index).copied().flatten().is_some() {
            continue;
        }
        let links = (0..inputs.source_seeds.len())
            .filter(|source| {
                inputs
                    .usable
                    .get(source * mics + index)
                    .copied()
                    .unwrap_or(false)
            })
            .count();
        if links < MIN_RANGES_FOR_FIX {
            notes.push(format!(
                "joint self-calibration not attempted: microphone {} has {links} usable links (< {MIN_RANGES_FOR_FIX} required); direct solves stand",
                microphone.id
            ));
        }
    }
    notes
}

/// Blank refused source entry for a failed joint attempt (Q10b).
fn joint_refused_source(source_id: &str) -> SourcePosition {
    SourcePosition {
        source_id: source_id.into(),
        position_m: None,
        sigma_m: None,
        rms_residual_m: None,
        gdop: None,
        mirror_ambiguous: false,
        declared_error_m: None,
        loo_max_residual_m: None,
        loo_worst_microphone: None,
        loo_max_shift_m: None,
        reason: Some("solve_failed".into()),
    }
}

/// Blank refused microphone entry for a failed joint attempt (Q10b).
fn joint_refused_microphone(microphone_id: &str) -> MicPosition {
    MicPosition {
        microphone_id: microphone_id.into(),
        position_m: None,
        sigma_m: None,
        rms_residual_m: None,
        gdop: None,
        mirror_ambiguous: false,
        declared_error_m: None,
        loo_max_residual_m: None,
        loo_worst_source: None,
        loo_max_shift_m: None,
        reason: Some("solve_failed".into()),
    }
}

/// Map one joint point into a source report entry.
///
/// GDOP stays empty (the joint covariance marginals replace the
/// single-point dilution figure) and LOO stays empty (the solver runs
/// no joint leave-one-out: held-out links poison verdicts through
/// frame flex); the global mirror flag applies to every jointly
/// estimated point.
fn joint_mapped_source(
    source_id: &str,
    point: &JointPoint,
    mirror_ambiguous: bool,
) -> SourcePosition {
    SourcePosition {
        source_id: source_id.into(),
        position_m: Some(point.position_m),
        sigma_m: Some(point.sigma_m),
        rms_residual_m: Some(point.rms_residual_m),
        gdop: None,
        mirror_ambiguous,
        declared_error_m: None,
        loo_max_residual_m: None,
        loo_worst_microphone: None,
        loo_max_shift_m: None,
        reason: None,
    }
}

/// Map one joint point into a microphone report entry (see
/// [`joint_mapped_source`] for the empty fields).
fn joint_mapped_microphone(
    microphone_id: &str,
    point: &JointPoint,
    mirror_ambiguous: bool,
) -> MicPosition {
    MicPosition {
        microphone_id: microphone_id.into(),
        position_m: Some(point.position_m),
        sigma_m: Some(point.sigma_m),
        rms_residual_m: Some(point.rms_residual_m),
        gdop: None,
        mirror_ambiguous,
        declared_error_m: None,
        loo_max_residual_m: None,
        loo_worst_source: None,
        loo_max_shift_m: None,
        reason: None,
    }
}

/// Jointly calibrate unsurveyed sources and microphones (§6.4).
///
/// `Err` carries Q10 session notes (counting failed: the caller falls
/// back to direct solves, which stand). `Ok` is the joint attempt —
/// positions, notes, and the calibration summary, or a `None` summary
/// when the attempt itself failed past counting (Q10b: every
/// unsurveyed point refused with `solve_failed`, ranges kept, while
/// surveyed audits still publish).
fn try_joint_positions(
    plan: &CaptureSessionPlan,
    ranges: &[SourceRanges],
    sound_speed_m_s: f64,
    sample_rate_hz: u32,
) -> Result<JointAttempt, Vec<String>> {
    let inputs = build_joint_inputs(plan, ranges);
    let q10 = joint_counting_notes(plan, &inputs);
    if !q10.is_empty() {
        return Err(q10);
    }
    let sample_m = sound_speed_m_s / f64::from(sample_rate_hz.max(1));
    let Some(estimate) = joint_self_calibrate(
        &inputs.source_seeds,
        &inputs.mic_seeds,
        &inputs.ranges_sm,
    ) else {
        let (sources, microphones, notes) = map_joint_refused(plan, ranges, sample_m);
        return Ok(JointAttempt {
            sources,
            microphones,
            notes,
            calibration: None,
        });
    };
    let mut notes = Vec::new();
    let sources = map_joint_sources(plan, ranges, &estimate, sample_m, &mut notes);
    let microphones = map_joint_microphones(plan, ranges, &estimate, sample_m, &mut notes);
    notes.extend(joint_exclusion_notes(plan, &inputs, &estimate));
    if estimate.outlier_absorb_risk {
        notes.push(
            "joint self-calibration on thin surveyed coverage: a gross range outlier can absorb into the frame with clean residuals; verify the geometry when outliers are possible"
                .into(),
        );
    }
    let calibration = JointCalibration {
        sources_solved: inputs.source_seeds.iter().filter(|seed| seed.is_none()).count(),
        microphones_solved: inputs.mic_seeds.iter().filter(|seed| seed.is_none()).count(),
        anchors: inputs
            .source_seeds
            .iter()
            .chain(inputs.mic_seeds.iter())
            .filter(|seed| seed.is_some())
            .count(),
        rms_residual_m: estimate.rms_residual_m,
        mirror_ambiguous: estimate.mirror_ambiguous,
        outlier_absorb_risk: estimate.outlier_absorb_risk,
    };
    Ok(JointAttempt {
        sources,
        microphones,
        notes,
        calibration: Some(calibration),
    })
}

/// Map a failed joint attempt (Q10b): every unsurveyed point refused
/// with `solve_failed` while surveyed audits still publish.
fn map_joint_refused(
    plan: &CaptureSessionPlan,
    ranges: &[SourceRanges],
    sample_m: f64,
) -> (Vec<SourcePosition>, Vec<MicPosition>, Vec<String>) {
    let mut sources = Vec::with_capacity(ranges.len());
    let mut microphones = Vec::with_capacity(plan.microphones.len());
    let mut notes = Vec::new();
    for row in ranges {
        let surveyed = plan
            .sources
            .iter()
            .find(|source| source.id == row.source_id)
            .and_then(|source| source.position_m);
        if let Some(declared) = surveyed {
            let (position, mut audit) =
                solve_row_position(row, &plan.microphones, Some(declared), sample_m);
            sources.push(position);
            notes.append(&mut audit);
        } else {
            notes.push(format!(
                "source {} position refused: joint self-calibration failed (solve_failed)",
                row.source_id
            ));
            sources.push(joint_refused_source(&row.source_id));
        }
    }
    for (index, microphone) in plan.microphones.iter().enumerate() {
        if microphone.position_m.is_some() {
            let (position, mut audit) = solve_column_position(
                &microphone.id,
                index,
                ranges,
                &plan.sources,
                microphone.position_m,
                sample_m,
            );
            microphones.push(position);
            notes.append(&mut audit);
        } else {
            notes.push(format!(
                "microphone {} position refused: joint self-calibration failed (solve_failed)",
                microphone.id
            ));
            microphones.push(joint_refused_microphone(&microphone.id));
        }
    }
    (sources, microphones, notes)
}

/// Map joint source estimates into report entries, auditing surveyed
/// rows forward as in direct mode.
fn map_joint_sources(
    plan: &CaptureSessionPlan,
    ranges: &[SourceRanges],
    estimate: &JointEstimate,
    sample_m: f64,
    notes: &mut Vec<String>,
) -> Vec<SourcePosition> {
    let mut sources = Vec::with_capacity(ranges.len());
    for row in ranges {
        let found = plan
            .sources
            .iter()
            .enumerate()
            .find(|(_, source)| source.id == row.source_id);
        let Some((index, source)) = found else {
            sources.push(joint_refused_source(&row.source_id));
            continue;
        };
        if let Some(declared) = source.position_m {
            let (position, mut audit) =
                solve_row_position(row, &plan.microphones, Some(declared), sample_m);
            sources.push(position);
            notes.append(&mut audit);
            continue;
        }
        let point = estimate
            .source_points
            .get(index)
            .and_then(|point| point.as_ref());
        if let Some(point) = point {
            sources.push(joint_mapped_source(
                &source.id,
                point,
                estimate.mirror_ambiguous,
            ));
        } else {
            notes.push(format!(
                "source {} position refused: joint self-calibration failed (solve_failed)",
                source.id
            ));
            sources.push(joint_refused_source(&source.id));
        }
    }
    sources
}

/// Map joint microphone estimates into report entries; surveyed
/// microphones stay anchors.
fn map_joint_microphones(
    plan: &CaptureSessionPlan,
    ranges: &[SourceRanges],
    estimate: &JointEstimate,
    sample_m: f64,
    notes: &mut Vec<String>,
) -> Vec<MicPosition> {
    let mut microphones = Vec::with_capacity(plan.microphones.len());
    for (index, microphone) in plan.microphones.iter().enumerate() {
        if let Some(surveyed) = microphone.position_m {
            let (position, mut audit) = solve_column_position(
                &microphone.id,
                index,
                ranges,
                &plan.sources,
                Some(surveyed),
                sample_m,
            );
            microphones.push(position);
            notes.append(&mut audit);
            continue;
        }
        let point = estimate
            .mic_points
            .get(index)
            .and_then(|point| point.as_ref());
        if let Some(point) = point {
            microphones.push(joint_mapped_microphone(
                &microphone.id,
                point,
                estimate.mirror_ambiguous,
            ));
        } else {
            notes.push(format!(
                "microphone {} position refused: joint self-calibration failed (solve_failed)",
                microphone.id
            ));
            microphones.push(joint_refused_microphone(&microphone.id));
        }
    }
    microphones
}

/// One session note per link the joint fit excluded as a range
/// outlier, in source-major cell order.
fn joint_exclusion_notes(
    plan: &CaptureSessionPlan,
    inputs: &JointInputs,
    estimate: &JointEstimate,
) -> Vec<String> {
    let mut notes = Vec::new();
    let mics = plan.microphones.len();
    for (cell, used) in estimate.used_links.iter().enumerate() {
        if *used || !inputs.usable.get(cell).copied().unwrap_or(false) {
            continue;
        }
        let (Some(source), Some(mic)) = (cell.checked_div(mics), cell.checked_rem(mics)) else {
            continue;
        };
        let (Some(source), Some(microphone)) =
            (plan.sources.get(source), plan.microphones.get(mic))
        else {
            continue;
        };
        notes.push(format!(
            "source {}: {} excluded from the joint fit as a range outlier",
            source.id, microphone.id
        ));
    }
    notes
}

/// Outcome of the shared multilateration core (§6.1 + Q7).
///
/// Forward rows and inverse columns run the same solver with the same
/// gates and refusal tokens; only the anchor side differs. The row
/// and column assemblers map verdicts to their own report structs and
/// note wording.
enum PointVerdict {
    /// Solved; the assembler maps excluded slots to anchor names.
    Solved {
        estimate: PositionEstimate,
        excluded_slots: Vec<usize>,
    },
    /// Refused; the assembler formats notes from the evidence.
    Refused {
        reason: &'static str,
        mirror_ambiguous: bool,
        rms_residual_m: Option<f64>,
        gdop: Option<f64>,
        residual_gate_m: Option<f64>,
    },
}

/// Multilaterate one point from usable anchor cells.
///
/// `usable` pairs each cell's original index with its anchor position,
/// distance, and sigma. `None` from the solver means depletion or
/// divergence (`solve_failed`), never an input disagreement (see
/// [`gather_usable_cells`]).
fn solve_point(usable: &[(usize, [f64; 3], f64, f64)], sample_m: f64) -> PointVerdict {
    let refused = |reason: &'static str,
                   mirror_ambiguous: bool,
                   rms_residual_m: Option<f64>,
                   gdop: Option<f64>,
                   residual_gate_m: Option<f64>| {
        PointVerdict::Refused {
            reason,
            mirror_ambiguous,
            rms_residual_m,
            gdop,
            residual_gate_m,
        }
    };
    if usable.len() < MIN_RANGES_FOR_FIX {
        if usable.len() == 3 {
            let positions: Vec<[f64; 3]> =
                usable.iter().map(|(_, position, _, _)| *position).collect();
            if geometry_rank(&positions).collinear {
                return refused("degenerate_geometry", false, None, None, None);
            }
            return refused("mirror_ambiguous", true, None, None, None);
        }
        return refused("insufficient_ranges", false, None, None, None);
    }
    let positions: Vec<[f64; 3]> = usable.iter().map(|(_, position, _, _)| *position).collect();
    if geometry_rank(&positions).collinear {
        return refused("degenerate_geometry", false, None, None, None);
    }
    let ranges: Vec<Option<(f64, f64)>> = usable
        .iter()
        .map(|(_, _, distance, sigma)| Some((*distance, *sigma)))
        .collect();
    let Some(estimate) = solve_position(&positions, &ranges) else {
        return refused("solve_failed", false, None, None, None);
    };
    let excluded_slots: Vec<usize> = estimate
        .used
        .iter()
        .enumerate()
        .filter(|(_, used)| !**used)
        .map(|(slot, _)| slot)
        .collect();
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
        return refused(
            "solve_residual_high",
            estimate.mirror_ambiguous,
            Some(estimate.rms_residual_m),
            Some(estimate.gdop),
            Some(residual_gate_m),
        );
    }
    PointVerdict::Solved {
        estimate,
        excluded_slots,
    }
}

/// Multilaterate one source row with geometry and residual gates.
///
/// Thin assembler over [`solve_point`]: refusals set `reason` and
/// leave the position empty; `solve_residual_high` still publishes
/// the RMS and GDOP of the refused fit as evidence.
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
        loo_max_residual_m: None,
        loo_worst_microphone: None,
        loo_max_shift_m: None,
        reason: reason.map(str::to_string),
    };
    let mut notes = Vec::new();
    let usable = gather_usable_cells(row, microphones);
    let anchor_ids: Vec<String> = usable
        .iter()
        .map(|(index, _, _, _)| row.cells[*index].microphone_id.clone())
        .collect();
    match solve_point(&usable, sample_m) {
        PointVerdict::Refused {
            reason,
            mirror_ambiguous,
            rms_residual_m,
            gdop,
            residual_gate_m,
        } => {
            if reason == "solve_failed" {
                notes.push(format!(
                    "source {} position refused: multilateration failed (solve_failed)",
                    row.source_id
                ));
            }
            let mut refused = blank(Some(reason), mirror_ambiguous);
            if reason == "solve_residual_high" {
                if let (Some(rms), Some(gate)) = (rms_residual_m, residual_gate_m) {
                    notes.push(format!(
                        "source {} position refused: RMS residual {:.1} mm exceeds the {:.1} mm gate (solve_residual_high)",
                        row.source_id,
                        rms * 1000.0,
                        gate * 1000.0,
                    ));
                }
                refused.rms_residual_m = rms_residual_m;
                refused.gdop = gdop;
            }
            (refused, notes)
        }
        PointVerdict::Solved {
            estimate,
            excluded_slots,
        } => {
            for slot in excluded_slots {
                notes.push(format!(
                    "source {}: {} excluded from the position fit as a range outlier",
                    row.source_id, anchor_ids[slot]
                ));
            }
            let declared_error_m =
                declared_m.map(|surveyed| euclidean_m(surveyed, estimate.position_m));
            let positions: Vec<[f64; 3]> =
                usable.iter().map(|(_, position, _, _)| *position).collect();
            let ranges: Vec<Option<(f64, f64)>> = usable
                .iter()
                .map(|(_, _, distance, sigma)| Some((*distance, *sigma)))
                .collect();
            let (loo_max_residual_m, loo_worst_microphone, loo_max_shift_m) =
                loo_fields(&anchor_ids, &positions, &ranges, &estimate);
            let position = SourcePosition {
                source_id: row.source_id.clone(),
                position_m: Some(estimate.position_m),
                sigma_m: Some(estimate.sigma_m),
                rms_residual_m: Some(estimate.rms_residual_m),
                gdop: Some(estimate.gdop),
                mirror_ambiguous: estimate.mirror_ambiguous,
                declared_error_m,
                loo_max_residual_m,
                loo_worst_microphone,
                loo_max_shift_m,
                reason: None,
            };
            (position, notes)
        }
    }
}

/// Multilaterate one microphone column against surveyed sources.
///
/// Inverse assembler over [`solve_point`]: the same solver, gates,
/// and tokens as forward, with source anchors. Surveyed microphones
/// are anchors, never estimates — the solve is not attempted and the
/// entry carries `surveyed_anchor`.
fn solve_column_position(
    microphone_id: &str,
    microphone_index: usize,
    ranges: &[SourceRanges],
    sources: &[CaptureSource],
    surveyed_m: Option<[f64; 3]>,
    sample_m: f64,
) -> (MicPosition, Vec<String>) {
    let blank = |reason: Option<&str>, mirror_ambiguous: bool| MicPosition {
        microphone_id: microphone_id.to_string(),
        position_m: None,
        sigma_m: None,
        rms_residual_m: None,
        gdop: None,
        mirror_ambiguous,
        declared_error_m: None,
        loo_max_residual_m: None,
        loo_worst_source: None,
        loo_max_shift_m: None,
        reason: reason.map(str::to_string),
    };
    let mut notes = Vec::new();
    if surveyed_m.is_some() {
        return (blank(Some("surveyed_anchor"), false), notes);
    }
    let usable = gather_usable_column(microphone_index, ranges, sources);
    let anchor_ids: Vec<String> = usable
        .iter()
        .map(|(index, _, _, _)| ranges[*index].source_id.clone())
        .collect();
    match solve_point(&usable, sample_m) {
        PointVerdict::Refused {
            reason,
            mirror_ambiguous,
            rms_residual_m,
            gdop,
            residual_gate_m,
        } => {
            if reason == "solve_failed" {
                notes.push(format!(
                    "microphone {microphone_id} position refused: multilateration failed (solve_failed)"
                ));
            }
            let mut refused = blank(Some(reason), mirror_ambiguous);
            if reason == "solve_residual_high" {
                if let (Some(rms), Some(gate)) = (rms_residual_m, residual_gate_m) {
                    notes.push(format!(
                        "microphone {microphone_id} position refused: RMS residual {:.1} mm exceeds the {:.1} mm gate (solve_residual_high)",
                        rms * 1000.0,
                        gate * 1000.0,
                    ));
                }
                refused.rms_residual_m = rms_residual_m;
                refused.gdop = gdop;
            }
            (refused, notes)
        }
        PointVerdict::Solved {
            estimate,
            excluded_slots,
        } => {
            for slot in excluded_slots {
                notes.push(format!(
                    "microphone {microphone_id}: {} excluded from the position fit as a range outlier",
                    anchor_ids[slot]
                ));
            }
            let positions: Vec<[f64; 3]> =
                usable.iter().map(|(_, position, _, _)| *position).collect();
            let cell_ranges: Vec<Option<(f64, f64)>> = usable
                .iter()
                .map(|(_, _, distance, sigma)| Some((*distance, *sigma)))
                .collect();
            let (loo_max_residual_m, loo_worst_source, loo_max_shift_m) =
                loo_fields(&anchor_ids, &positions, &cell_ranges, &estimate);
            let position = MicPosition {
                microphone_id: microphone_id.to_string(),
                position_m: Some(estimate.position_m),
                sigma_m: Some(estimate.sigma_m),
                rms_residual_m: Some(estimate.rms_residual_m),
                gdop: Some(estimate.gdop),
                mirror_ambiguous: estimate.mirror_ambiguous,
                declared_error_m: None,
                loo_max_residual_m,
                loo_worst_source,
                loo_max_shift_m,
                reason: None,
            };
            (position, notes)
        }
    }
}

/// Gather solver-usable cells pairing range rows with plan microphones.
///
/// Cells pair by order (the builders' contract) and by id; a
/// misaligned cell counts as missing rather than misattributed. The
/// validity rules mirror the solver's exactly, so the geometry
/// pre-check and the solver classify the same subset.
fn gather_usable_cells(
    row: &SourceRanges,
    microphones: &[CaptureMicrophone],
) -> Vec<(usize, [f64; 3], f64, f64)> {
    let mut usable = Vec::new();
    for (index, cell) in row.cells.iter().enumerate() {
        let Some(microphone) = microphones.get(index) else {
            continue;
        };
        if cell.microphone_id != microphone.id {
            continue;
        }
        // Unsurveyed microphones keep their ranges but cannot anchor a
        // forward solve; they wait for the inverse or joint solve.
        let Some(anchor) = microphone.position_m else {
            continue;
        };
        if let (Some(distance), Some(sigma)) = (cell.distance_m, cell.sigma_m) {
            if distance.is_finite() && distance >= 0.0 && sigma.is_finite() && sigma > 0.0 {
                usable.push((index, anchor, distance, sigma));
            }
        }
    }
    usable
}

/// Leave-one-out diagnostics for a solved row or column (§7.2 step 8).
///
/// Re-solves per used anchor with that range dropped and reports the
/// largest held-out residual (with its anchor label) and the largest
/// solution shift. `anchor_ids` runs parallel to `positions`/`ranges`.
/// Minimum geometry (four used ranges) diagnoses nothing — every drop
/// depletes — and reports `None`, which is honest, not missing.
fn loo_fields(
    anchor_ids: &[String],
    positions: &[[f64; 3]],
    ranges: &[Option<(f64, f64)>],
    estimate: &PositionEstimate,
) -> (Option<f64>, Option<String>, Option<f64>) {
    match loo_diagnostics(positions, ranges, estimate) {
        Some(diagnostics) => (
            Some(diagnostics.max_residual_m),
            anchor_ids.get(diagnostics.worst_index).cloned(),
            Some(diagnostics.max_shift_m),
        ),
        None => (None, None, None),
    }
}

/// Gather solver-usable cells down one microphone column, pairing range
/// rows with surveyed plan sources.
///
/// Rows pair with plan sources by order and by id; a misaligned row
/// counts as missing rather than misattributed. Only surveyed sources
/// anchor an inverse solve — unsurveyed sources wait for the joint
/// solve. The validity rules mirror the solver's exactly.
fn gather_usable_column(
    microphone_index: usize,
    ranges: &[SourceRanges],
    sources: &[CaptureSource],
) -> Vec<(usize, [f64; 3], f64, f64)> {
    let mut usable = Vec::new();
    for (index, row) in ranges.iter().enumerate() {
        let Some(source) = sources.get(index) else {
            continue;
        };
        if row.source_id != source.id {
            continue;
        }
        let Some(cell) = row.cells.get(microphone_index) else {
            continue;
        };
        let Some(surveyed) = source.position_m else {
            continue;
        };
        if let (Some(distance), Some(sigma)) = (cell.distance_m, cell.sigma_m) {
            if distance.is_finite() && distance >= 0.0 && sigma.is_finite() && sigma > 0.0 {
                usable.push((index, surveyed, distance, sigma));
            }
        }
    }
    usable
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
        .filter_map(|microphone| microphone.position_m)
        .collect();
    let surveyed = positions.len();
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
    if surveyed < microphones {
        lines.push(format!(
            "localization readiness: {}/{} microphones unsurveyed; forward solves use surveyed anchors only, the rest wait for inverse/joint solves",
            microphones - surveyed,
            microphones
        ));
    }
    if surveyed < MIN_RANGES_FOR_FIX {
        lines.push(format!(
            "localization readiness: {surveyed}/4 surveyed microphones, forward positions need at least 4"
        ));
    }
    if surveyed >= MIN_RANGES_FOR_FIX {
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
    // Microphone lines print only when some microphone is unsurveyed;
    // all-surveyed sessions keep the Phase 3 output byte-identical.
    if report
        .microphones
        .iter()
        .any(|mic| mic.reason.as_deref() != Some("surveyed_anchor"))
    {
        for microphone in &report.microphones {
            lines.push(microphone_summary_line(microphone));
        }
    }
    if let Some(calibration) = &report.joint {
        lines.push(joint_summary_line(calibration));
    }
    lines
}

/// One CLI line for a joint self-calibration (§8.1).
fn joint_summary_line(calibration: &JointCalibration) -> String {
    let mut detail = String::new();
    if calibration.mirror_ambiguous {
        detail.push_str(" [mirror-ambiguous]");
    }
    if calibration.outlier_absorb_risk {
        detail.push_str(" [outlier-absorb-risk]");
    }
    format!(
        "localization joint: {} sources + {} microphones from {} anchors, rms {:.1} mm{}",
        calibration.sources_solved,
        calibration.microphones_solved,
        calibration.anchors,
        calibration.rms_residual_m * 1000.0,
        detail
    )
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

/// One microphone position line, mirroring [`position_summary_line`].
fn microphone_summary_line(position: &MicPosition) -> String {
    let microphone_id = &position.microphone_id;
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
        format!("localization mic {microphone_id} position: [{x:.3}, {y:.3}, {z:.3}] m{detail}")
    } else {
        let reason = position.reason.as_deref().unwrap_or("unknown");
        format!("localization mic {microphone_id} position: refused ({reason}){detail}")
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
            (propagation_to_meters(480.0, 48_000.0, 343.0, source, 0.0).unwrap() - 3.43).abs()
                < 1e-9
        );
        let delayed = crate::capture_session::CaptureSource {
            source_delay_calibration_ms: Some(1.0),
            ..source.clone()
        };
        assert!(
            (propagation_to_meters(480.0, 48_000.0, 343.0, &delayed, 0.0).unwrap()
                - (3.43 - 0.343))
                .abs()
                < 1e-9
        );
        assert!(
            (propagation_to_meters(480.0, 48_000.0, 343.0, source, 0.01).unwrap() - (3.43 - 0.01))
                .abs()
                < 1e-9
        );
        assert_eq!(
            propagation_to_meters(f64::NAN, 48_000.0, 343.0, source, 0.0),
            None
        );
        let sigma = range_sigma_m(1.0, 3.43, 48_000.0, 343.0, 0.5, 0.0).unwrap();
        assert!(sigma > 343.0 / 48_000.0 && sigma < 0.05);
        let biased = range_sigma_m(1.0, 3.43, 48_000.0, 343.0, 0.5, BENCH_BIAS_RESIDUAL_M).unwrap();
        assert!((biased - sigma - BENCH_BIAS_RESIDUAL_M).abs() < 1e-12);
        assert_eq!(
            range_sigma_m(f64::INFINITY, 3.43, 48_000.0, 343.0, 0.5, 0.0),
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

    /// Rich geometry for Phase 4 inverse/joint tests: 5 non-coplanar
    /// sources by 6 non-coplanar microphones (the base golden's two
    /// sources cannot anchor an inverse solve).
    fn rich_golden() -> Golden {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/localization-golden-rich.json"
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
        /// Loopback plus surveyed emitter: the anchor configuration the
        /// latency cross-check needs for absolute clock origins.
        Anchor,
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

    /// Build a complete synthetic single-repeat session for one mode.
    fn harness(mode: HarnessMode) -> Harness {
        harness_repeats(mode, 1, None, None, golden())
    }

    /// Rich-geometry session for Phase 4 inverse/joint tests.
    fn harness_rich(mode: HarnessMode) -> Harness {
        harness_repeats(mode, 1, None, None, rich_golden())
    }

    /// Synthesize one harness take with its clock fit into the manifest.
    ///
    /// `wild_extra` shifts only the sweep path, leaving chirps clean so
    /// the clock still fits and the median must reject the outlier.
    #[expect(
        clippy::too_many_arguments,
        reason = "The golden harness threads synthesis inputs explicitly; bundling would hide the per-take parameters."
    )]
    fn synth_harness_take(
        mode: HarnessMode,
        golden: &Golden,
        stimulus: &crate::capture_session::protocol::CaptureStimulus,
        take_stimulus: &[f32],
        layout: &crate::capture_session::protocol::CaptureStimulusLayout,
        rate: u32,
        rate_f64: f64,
        chirp_source: [f64; 3],
        source_id: &str,
        source_position: [f64; 3],
        mic_id: &str,
        mic_position: [f64; 3],
        wild_extra: f64,
        repeat: u32,
        manifest: &mut RawCaptureManifest,
        takes: &mut Vec<crate::capture_session::clock::io::ClockProcessedTake>,
        fits: &mut Vec<Option<crate::capture_session::clock::CaptureClockFit>>,
        output: &tempfile::TempDir,
        audio_index: &mut usize,
    ) {
        let src_prop = distance_meter(source_position, mic_position) / golden.sound_speed_m_s
            * rate_f64
            + wild_extra;
        let chirp_prop = if mode == HarnessMode::Emitter || mode == HarnessMode::Anchor {
            distance_meter(golden.timing_emitter_m, mic_position) / golden.sound_speed_m_s
                * rate_f64
        } else {
            distance_meter(chirp_source, mic_position) / golden.sound_speed_m_s * rate_f64
        };
        let samples = synth_take(
            take_stimulus,
            layout,
            golden.session_clock.offset_samples,
            golden.session_clock.skew_ppm,
            golden.latency_samples,
            src_prop,
            chirp_prop,
        );
        let wav_file = format!("take-{source_id}-{mic_id}-r{repeat:03}-raw.wav");
        crate::signal_recorder::write_wav_file(&output.path().join(&wav_file), &samples, rate, 1)
            .unwrap();
        let take = crate::capture_session::record::RawCaptureTake {
            take_id: crate::capture_session::record::stable_take_id(source_id, mic_id, repeat),
            repeat_index: repeat,
            device_id: "id:iface".into(),
            output_device_id: "id:iface".into(),
            source_id: source_id.into(),
            microphone_id: mic_id.into(),
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
            manifest,
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
        *audio_index += 1;
    }

    /// Synthesize one harness loopback file for a source repeat.
    #[expect(
        clippy::too_many_arguments,
        reason = "The golden harness threads synthesis inputs explicitly; bundling would hide the per-repeat parameters."
    )]
    fn synth_harness_loopback(
        golden: &Golden,
        stimulus: &crate::capture_session::protocol::CaptureStimulus,
        layout: &crate::capture_session::protocol::CaptureStimulusLayout,
        rate: u32,
        raw_root: &tempfile::TempDir,
        manifest: &mut RawCaptureManifest,
        source_id: &str,
        repeat: u32,
    ) {
        let samples = synth_take(
            &stimulus.samples,
            layout,
            golden.session_clock.offset_samples,
            golden.session_clock.skew_ppm,
            golden.latency_samples,
            0.0,
            0.0,
        );
        let wav_file = format!("take-{source_id}-loopback-r{repeat:03}-raw.wav");
        crate::signal_recorder::write_wav_file(&raw_root.path().join(&wav_file), &samples, rate, 1)
            .unwrap();
        manifest
            .loopback_files
            .push(crate::capture_session::record::CaptureLoopbackTake {
                repeat_index: repeat,
                source_id: source_id.into(),
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

    /// Build a synthetic session with several repeats and an optional
    /// wild arrival.
    ///
    /// `wild` is `(source id, microphone id, repeat, extra sweep delay
    /// in samples)` injected into one take's source path; its chirps
    /// stay clean so the clock still fits and the median must reject
    /// the outlier at combination time.
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
    fn harness_repeats(
        mode: HarnessMode,
        repeats: u32,
        wild: Option<(&str, &str, u32, f64)>,
        sweep_override: Option<(f64, f64)>,
        golden: Golden,
    ) -> Harness {
        let rate = golden.sample_rate_hz;
        let mut plan: CaptureSessionPlan =
            serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json"))
                .unwrap();
        plan.sweep.duration_secs = 0.25;
        // A sub measurement session plans a genuine low-frequency
        // sweep: the estimator can only correlate content the planned
        // sweep actually contains, and the timing chirps stay
        // full-band markers either way, so the clock still fits.
        if let Some((start_hz, end_hz)) = sweep_override {
            plan.sweep.start_hz = start_hz;
            plan.sweep.end_hz = end_hz;
        }
        plan.repeat_count = repeats;
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
                mic.position_m = Some(*position);
                mic.position_uncertainty_mm = Some(0.1);
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
        if mode == HarnessMode::Emitter || mode == HarnessMode::Anchor {
            plan.timing_reference = Some(CaptureTimingReference {
                output_channel: 2,
                position_m: golden.timing_emitter_m,
                position_uncertainty_mm: 0.1,
                sound_speed_m_s: golden.sound_speed_m_s,
                sound_speed_uncertainty_m_s: 0.1,
            });
        }
        if mode == HarnessMode::Loopback || mode == HarnessMode::Anchor {
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
            timing_reference_output_channel: Some(
                if mode == HarnessMode::Emitter || mode == HarnessMode::Anchor {
                    2
                } else {
                    0
                },
            ),
            status: RawCaptureStatus::RawComplete,
            pending_processing: Vec::new(),
            calibrations: Vec::new(),
            takes: Vec::new(),
            error: None,
        };
        let mut takes = Vec::new();
        let mut fits = Vec::new();
        let mut audio_index = 0_usize;
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
                for repeat in 0..repeats {
                    let wild_extra = match wild {
                        Some((source, mic, wild_repeat, extra))
                            if source == source_id.as_str()
                                && mic == mic_id.as_str()
                                && wild_repeat == repeat =>
                        {
                            extra
                        }
                        _ => 0.0,
                    };
                    synth_harness_take(
                        mode,
                        &golden,
                        &stimulus,
                        stimulus.samples.as_slice(),
                        &layout,
                        rate,
                        rate_f64,
                        chirp_source,
                        source_id,
                        *source_position,
                        mic_id,
                        *mic_position,
                        wild_extra,
                        repeat,
                        &mut manifest,
                        &mut takes,
                        &mut fits,
                        &output,
                        &mut audio_index,
                    );
                }
            }
            if mode == HarnessMode::Loopback || mode == HarnessMode::Anchor {
                for repeat in 0..repeats {
                    synth_harness_loopback(
                        &golden,
                        &stimulus,
                        &layout,
                        rate,
                        &raw_root,
                        &mut manifest,
                        source_id,
                        repeat,
                    );
                }
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
            // Four microphones leave no spare cell to hold out.
            assert_eq!(position.loo_max_residual_m, None);
            assert_eq!(position.loo_worst_microphone, None);
            assert_eq!(position.loo_max_shift_m, None);
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
                microphone.position_m = Some(*position);
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
    fn combine_single_repeat_passes_through_exactly() {
        let truth = [2.3, -1.1, 0.7];
        let (microphones, row) = tetra_row(truth, 0.001);
        let plan = position_plan(&microphones, None);
        let combined = combine_repeat_rows(&plan, &plan.sources[0], std::slice::from_ref(&row));
        assert_eq!(combined, row);
    }

    #[test]
    fn combine_repeat_rows_takes_median_and_widens_sigma() {
        let microphones = vec![("mic-0", [0.0, 0.0, 0.0])];
        let plan = position_plan(&microphones, None);
        let mut rows = Vec::new();
        for (distance, arrival, confidence) in
            [(2.0, 100.0, 20.0), (2.5, 105.0, 18.0), (2.01, 101.0, 22.0)]
        {
            let mut row = position_row("left", vec![("mic-0", Some((distance, 0.001)))]);
            row.cells[0].arrival_samples = Some(arrival);
            row.cells[0].reference_arrival_samples = Some(arrival - 50.0);
            row.cells[0].confidence_db = Some(confidence);
            row.cells[0].band = Some("sweep_2000_8000".into());
            rows.push(row);
        }
        let combined = combine_repeat_rows(&plan, &plan.sources[0], &rows);
        let cell = &combined.cells[0];
        assert!((cell.distance_m.unwrap_or(f64::NAN) - 2.01).abs() < 1.0e-12);
        assert!((cell.arrival_samples.unwrap_or(f64::NAN) - 101.0).abs() < 1.0e-12);
        assert!((cell.reference_arrival_samples.unwrap_or(f64::NAN) - 51.0).abs() < 1.0e-12);
        assert_eq!(cell.confidence_db, Some(18.0));
        assert_eq!(cell.band.as_deref(), Some("sweep_2000_8000"));
        assert_eq!(cell.reason, None);
        // Median estimator sigma widened by half the 0.5 m spread.
        let sigma = cell.sigma_m.unwrap_or(f64::NAN);
        assert!((sigma - 0.250_002).abs() < 1.0e-6, "sigma {sigma}");
    }

    #[test]
    fn combine_all_refused_keeps_common_token() {
        let microphones = vec![("mic-0", [0.0, 0.0, 0.0])];
        let plan = position_plan(&microphones, None);
        let mut rows = Vec::new();
        for token in ["clock_outlier", "clock_outlier", "arrival_ambiguous"] {
            let mut row = position_row("left", vec![("mic-0", None)]);
            row.cells[0].reason = Some(token.into());
            rows.push(row);
        }
        let combined = combine_repeat_rows(&plan, &plan.sources[0], &rows);
        let cell = &combined.cells[0];
        assert_eq!(cell.distance_m, None);
        assert_eq!(cell.reason.as_deref(), Some("clock_outlier"));
    }

    #[test]
    fn combine_empty_rows_refuses_take_missing() {
        let microphones = vec![("mic-0", [0.0, 0.0, 0.0])];
        let plan = position_plan(&microphones, None);
        let combined = combine_repeat_rows(&plan, &plan.sources[0], &[]);
        assert_eq!(combined.cells.len(), 1);
        assert_eq!(combined.cells[0].distance_m, None);
        assert_eq!(combined.cells[0].reason.as_deref(), Some("take_missing"));
    }

    #[test]
    fn combine_relative_rows_medians_tdoa() {
        let microphones = vec![("mic-0", [0.0, 0.0, 0.0])];
        let plan = position_plan(&microphones, None);
        let mut rows = Vec::new();
        for (tdoa, sigma) in [(10.0, 0.5), (12.0, 0.5)] {
            let mut row = position_row("left", vec![("mic-0", None)]);
            row.cells[0].tdoa_samples_vs_first = Some(tdoa);
            row.cells[0].tdoa_sigma_samples = Some(sigma);
            row.cells[0].reason = Some("no_absolute_time_origin".into());
            rows.push(row);
        }
        let combined = combine_repeat_rows(&plan, &plan.sources[0], &rows);
        let cell = &combined.cells[0];
        assert_eq!(cell.distance_m, None);
        assert_eq!(cell.sigma_m, None);
        assert!((cell.tdoa_samples_vs_first.unwrap_or(f64::NAN) - 11.0).abs() < 1.0e-12);
        let sigma = cell.tdoa_sigma_samples.unwrap_or(f64::NAN);
        assert!((sigma - 1.118_034).abs() < 1.0e-6, "sigma {sigma}");
        assert_eq!(cell.reason.as_deref(), Some("no_absolute_time_origin"));
    }

    #[test]
    fn repeat_median_rejects_wild_arrival() {
        let probe = golden();
        let wild_extra = 0.5 / probe.sound_speed_m_s * f64::from(probe.sample_rate_hz);
        let harness = harness_repeats(
            HarnessMode::Loopback,
            3,
            Some(("left", "mic-1", 1, wild_extra)),
            None,
            golden(),
        );
        let localized = localize_session(
            &harness.report,
            harness.raw_root.path(),
            harness.output.path(),
            &harness.fits,
        );
        assert_eq!(localized.mode, LocalizationMode::LoopbackReferenced);
        let row = localized
            .ranges
            .iter()
            .find(|row| row.source_id == "left")
            .expect("left row");
        let cell = row
            .cells
            .iter()
            .find(|cell| cell.microphone_id == "mic-1")
            .expect("mic-1 cell");
        let expected = harness.golden.expected_ranges_m["left/mic-1"];
        let distance = cell.distance_m.unwrap_or(f64::NAN);
        assert!(
            (distance - expected).abs() < 0.002,
            "median should reject the wild repeat: got {distance}, want {expected}"
        );
        // The wild repeat still widens sigma through the spread term.
        let sigma = cell.sigma_m.unwrap_or(f64::NAN);
        assert!(sigma > 0.1, "sigma {sigma}");
        assert_eq!(cell.reason, None);
        // The median row still solves a position near truth.
        let position = localized
            .sources
            .iter()
            .find(|source| source.source_id == "left")
            .expect("left position");
        let fit = position.position_m.expect("median row solves");
        let truth = harness.golden.sources_m["left"];
        assert!(distance_meter(fit, truth) < 0.02);
    }

    fn write_bench(dir: &tempfile::TempDir, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    const BENCH_VALID: &str = r#"{
        "schema": "sotf-capture/localization-calibration",
        "schema_version": 1,
        "interface": "iface",
        "microphone_model": "mic",
        "latency_samples_by_rate_hz": {"48000": 213.0},
        "mic_bias_correction_m": 0.004,
        "mic_bias_measured": true
    }"#;

    #[test]
    fn sub_take_localizes_in_low_band() {
        // A sub measurement session plans a genuine 20-300 Hz sweep
        // while the timing chirps stay full-band markers, so the
        // clock still fits and every sweep arrival goes low-band.
        let harness = harness_repeats(
            HarnessMode::Loopback,
            1,
            None,
            Some((20.0, 300.0)),
            golden(),
        );
        let localized = localize_session(
            &harness.report,
            harness.raw_root.path(),
            harness.output.path(),
            &harness.fits,
        );
        assert_eq!(localized.mode, LocalizationMode::LoopbackReferenced);
        for row in &localized.ranges {
            for cell in &row.cells {
                assert_eq!(cell.band.as_deref(), Some("sweep_20_300"));
                let key = format!("{}/{}", row.source_id, cell.microphone_id);
                let expected = harness.golden.expected_ranges_m[&key];
                let distance = cell.distance_m.unwrap_or(f64::NAN);
                assert!(
                    (distance - expected).abs() < 0.01,
                    "{key}: got {distance}, want {expected}"
                );
                // The low band resolves the hump but budgets honestly:
                // sigma stays near two decimetres while the synthetic
                // peak itself lands within millimetres.
                assert!(
                    cell.sigma_m.is_some_and(|sigma| sigma > 0.05),
                    "{key}: low-band sigma must stay wide, got {:?}",
                    cell.sigma_m
                );
            }
        }
        for source in &localized.sources {
            let fit = source.position_m.expect("low-band row solves");
            let truth = harness.golden.sources_m[source.source_id.as_str()];
            let error = distance_meter(fit, truth);
            assert!(error < 0.01, "low-band position error {error}");
        }
    }

    #[test]
    fn bench_loader_accepts_and_rejects() {
        let dir = tempfile::tempdir().unwrap();
        let calibration = load_bench_calibration(&write_bench(&dir, "bench.json", BENCH_VALID))
            .expect("valid bench file loads");
        assert_eq!(calibration.interface, "iface");
        assert_eq!(calibration.microphone_model, "mic");
        assert_eq!(calibration.loopback_lag_samples.get(&48_000), Some(&213.0));
        assert!((calibration.mic_bias_correction_m - 0.004).abs() < 1.0e-12);
        assert!(calibration.mic_bias_measured);
        let rejects = [
            (
                "bad schema",
                BENCH_VALID.replace("localization-calibration", "other"),
                "schema must be",
            ),
            (
                "bad version",
                BENCH_VALID.replace("\"schema_version\": 1", "\"schema_version\": 2"),
                "unsupported schema version",
            ),
            (
                "bad rate key",
                BENCH_VALID.replace("\"48000\"", "\"48k\""),
                "not a sample rate",
            ),
            (
                "negative lag",
                BENCH_VALID.replace("213.0", "-1.0"),
                "finite and nonnegative",
            ),
            (
                "overflowing bias",
                BENCH_VALID.replace("0.004", "1e400"),
                "cannot parse",
            ),
            (
                "measured without value",
                BENCH_VALID.replace("\"mic_bias_correction_m\": 0.004,", ""),
                "absent",
            ),
            ("malformed", "{oops".into(), "cannot parse"),
        ];
        for (name, content, fragment) in rejects {
            let error = load_bench_calibration(&write_bench(&dir, "bench.json", &content))
                .expect_err(&format!("{name} must fail"));
            assert!(
                error.contains(fragment),
                "{name}: {error} misses {fragment:?}"
            );
        }
        let vacuous = BENCH_VALID
            .replace("\"latency_samples_by_rate_hz\": {\"48000\": 213.0},", "")
            .replace(
                "\"mic_bias_measured\": true",
                "\"mic_bias_measured\": false",
            );
        let error = load_bench_calibration(&write_bench(&dir, "bench.json", &vacuous))
            .expect_err("vacuous bench file must fail");
        assert!(error.contains("neither"), "vacuous: {error}");
        let error = load_bench_calibration(&dir.path().join("missing.json"))
            .expect_err("missing bench file must fail");
        assert!(error.contains("cannot read"), "missing: {error}");
    }

    #[test]
    fn bench_bias_shifts_ranges() {
        let mut harness = harness(HarnessMode::Loopback);
        write_bench(
            &harness.raw_root,
            "bench.json",
            &BENCH_VALID
                .replace("\"latency_samples_by_rate_hz\": {\"48000\": 213.0},", "")
                .replace("0.004", "0.005"),
        );
        harness.report.plan.localization_calibration_file = Some("bench.json".into());
        let localize = |harness: &Harness| {
            localize_session(
                &harness.report,
                harness.raw_root.path(),
                harness.output.path(),
                &harness.fits,
            )
        };
        let localized = localize(&harness);
        assert!(
            (localized.mic_bias_correction_m - 0.005).abs() < 1.0e-12,
            "bias {}",
            localized.mic_bias_correction_m
        );
        for row in &localized.ranges {
            for cell in &row.cells {
                let key = format!("{}/{}", row.source_id, cell.microphone_id);
                let expected = harness.golden.expected_ranges_m[&key] - 0.005;
                let distance = cell.distance_m.unwrap_or(f64::NAN);
                assert!(
                    (distance - expected).abs() < 0.002,
                    "{key}: got {distance}, want {expected}"
                );
                assert!(
                    cell.sigma_m.unwrap_or(f64::NAN) > BENCH_BIAS_RESIDUAL_M,
                    "{key}: sigma must carry the residual budget"
                );
            }
        }
        assert!(
            localized
                .issues
                .iter()
                .all(|issue| !issue.contains("bench calibration unusable")),
            "unexpected bench issues: {:?}",
            localized.issues
        );
        // An unmeasured value applies nothing even when present.
        write_bench(
            &harness.raw_root,
            "bench.json",
            &BENCH_VALID.replace(
                "\"mic_bias_measured\": true",
                "\"mic_bias_measured\": false",
            ),
        );
        let localized = localize(&harness);
        assert_eq!(localized.mic_bias_correction_m, 0.0);
        for row in &localized.ranges {
            for cell in &row.cells {
                let key = format!("{}/{}", row.source_id, cell.microphone_id);
                let expected = harness.golden.expected_ranges_m[&key];
                let distance = cell.distance_m.unwrap_or(f64::NAN);
                assert!(
                    (distance - expected).abs() < 0.002,
                    "{key}: got {distance}, want {expected}"
                );
            }
        }
    }

    #[test]
    fn anchor_session_trusts_loopback_without_mismatch() {
        let harness = harness_repeats(HarnessMode::Anchor, 1, None, None, golden());
        for take in &harness.report.takes {
            assert_eq!(
                take.clock.basis,
                crate::capture_session::clock::CaptureClockBasis::FixedAcousticReference,
                "anchor takes need absolute origins"
            );
        }
        let localized = localize_session(
            &harness.report,
            harness.raw_root.path(),
            harness.output.path(),
            &harness.fits,
        );
        assert_eq!(localized.mode, LocalizationMode::LoopbackReferenced);
        assert!(
            localized
                .issues
                .iter()
                .all(|issue| !issue.contains("emitter_mismatch")),
            "surveyed emitter must agree: {:?}",
            localized.issues
        );
        for position in &localized.sources {
            let truth = harness.golden.sources_m[&position.source_id];
            let fit = position.position_m.expect("anchor ranges solve");
            assert!(distance_meter(fit, truth) < 0.02);
        }
    }

    #[test]
    fn inverse_session_solves_microphones_from_surveyed_sources() {
        // Pure inverse session: every source surveyed, no microphone
        // surveyed. Forward rows deplete (no anchors) while every
        // microphone column solves against the five surveyed sources.
        let mut harness = harness_rich(HarnessMode::Loopback);
        let source_truth = harness.golden.sources_m.clone();
        for source in &mut harness.report.plan.sources {
            source.position_m = source_truth.get(&source.id).copied();
            source.position_uncertainty_mm = Some(0.1);
        }
        for mic in &mut harness.report.plan.microphones {
            mic.position_m = None;
            mic.position_uncertainty_mm = None;
        }
        let localized = localize_session(
            &harness.report,
            harness.raw_root.path(),
            harness.output.path(),
            &harness.fits,
        );
        assert_eq!(localized.mode, LocalizationMode::LoopbackReferenced);
        // Ranges need no positions: the full matrix still publishes.
        for row in &localized.ranges {
            for cell in &row.cells {
                let key = format!("{}/{}", row.source_id, cell.microphone_id);
                let expected = harness.golden.expected_ranges_m[&key];
                let distance = cell.distance_m.unwrap_or(f64::NAN);
                assert!(
                    (distance - expected).abs() < 0.002,
                    "{key}: got {distance}, want {expected}"
                );
            }
        }
        // Forward solves cannot run without surveyed microphones.
        for position in &localized.sources {
            assert_eq!(position.position_m, None);
            assert_eq!(position.reason.as_deref(), Some("insufficient_ranges"));
        }
        // Every microphone solves against the surveyed sources.
        assert_eq!(localized.microphones.len(), 6);
        for microphone in &localized.microphones {
            let truth = harness.golden.microphones_m[&microphone.microphone_id];
            let fit = microphone.position_m.expect("inverse column solves");
            let error = distance_meter(fit, truth);
            assert!(error < 0.02, "{} error {error}", microphone.microphone_id);
            assert_eq!(microphone.reason, None);
            assert!(!microphone.mirror_ambiguous);
            // Five sources leave one spare for leave-one-out.
            assert!(microphone.loo_worst_source.is_some());
        }
        let lines = localization_summary_lines(&localized);
        assert!(
            lines
                .iter()
                .any(|line| line.contains("mic mic-0 position: ["))
        );
    }

    #[test]
    fn mixed_session_solves_forward_and_inverse_together() {
        // Four surveyed microphones anchor the forward solves; the two
        // remaining microphones solve inversely against surveyed
        // sources. Surveyed microphones are anchors, never estimates.
        let mut harness = harness_rich(HarnessMode::Loopback);
        let source_truth = harness.golden.sources_m.clone();
        for source in &mut harness.report.plan.sources {
            source.position_m = source_truth.get(&source.id).copied();
            source.position_uncertainty_mm = Some(0.1);
        }
        for mic in harness.report.plan.microphones.iter_mut().skip(4) {
            mic.position_m = None;
            mic.position_uncertainty_mm = None;
        }
        let localized = localize_session(
            &harness.report,
            harness.raw_root.path(),
            harness.output.path(),
            &harness.fits,
        );
        assert_eq!(localized.mode, LocalizationMode::LoopbackReferenced);
        for position in &localized.sources {
            let truth = harness.golden.sources_m[&position.source_id];
            let fit = position.position_m.expect("forward row solves");
            assert!(distance_meter(fit, truth) < 0.02);
            // Surveyed sources score their forward estimates.
            assert!(position.declared_error_m.is_some_and(|error| error < 0.02));
        }
        for microphone in &localized.microphones {
            if microphone.microphone_id == "mic-4" || microphone.microphone_id == "mic-5" {
                let truth = harness.golden.microphones_m[&microphone.microphone_id];
                let fit = microphone.position_m.expect("inverse column solves");
                assert!(distance_meter(fit, truth) < 0.02);
            } else {
                assert_eq!(microphone.position_m, None);
                assert_eq!(microphone.reason.as_deref(), Some("surveyed_anchor"));
            }
        }
    }

    /// Joint Q10b fixture: solver-probe geometry (5 sources, first
    /// two surveyed; 6 microphones, 0/1/2/4 surveyed) with exact 1 mm
    /// ranges except src-2, which keeps 4 links with a wild fourth —
    /// session counting passes but the solver depletes on exclusion.
    fn joint_q10b_fixture() -> (
        crate::capture_session::CaptureSessionPlan,
        Vec<SourceRanges>,
    ) {
        let sources: Vec<(&str, [f64; 3])> = vec![
            ("src-0", [3.5, 1.0, 1.4]),
            ("src-1", [3.5, 2.0, 1.0]),
            ("src-2", [1.0, -1.0, 1.6]),
            ("src-3", [1.0, 4.0, 0.9]),
            ("src-4", [1.0, 1.5, 2.4]),
        ];
        let mics: Vec<(&str, [f64; 3])> = vec![
            ("mic-0", [0.0, 0.0, 1.0]),
            ("mic-1", [2.0, 0.0, 1.2]),
            ("mic-2", [0.0, 2.0, 0.9]),
            ("mic-3", [2.0, 2.0, 1.5]),
            ("mic-4", [1.0, 1.0, 0.8]),
            ("mic-5", [1.0, 3.0, 1.1]),
        ];
        let mut plan: crate::capture_session::CaptureSessionPlan =
            serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json"))
                .unwrap();
        let template = plan.microphones[0].clone();
        plan.microphones = mics
            .iter()
            .enumerate()
            .map(|(channel, (id, position))| {
                let mut microphone = template.clone();
                microphone.id = (*id).to_string();
                microphone.input_channel = channel as u16;
                let surveyed = !matches!(channel, 3 | 5);
                microphone.position_m = surveyed.then_some(*position);
                microphone.position_uncertainty_mm = surveyed.then_some(0.1);
                microphone
            })
            .collect();
        plan.sources = sources
            .iter()
            .enumerate()
            .map(|(channel, (id, position))| {
                let surveyed = channel < 2;
                crate::capture_session::CaptureSource {
                    id: (*id).to_string(),
                    output_channel: channel as u16,
                    position_m: surveyed.then_some(*position),
                    position_uncertainty_mm: surveyed.then_some(0.1),
                    source_delay_calibration_ms: None,
                }
            })
            .collect();
        let ranges = sources
            .iter()
            .map(|(source_id, truth)| {
                position_row(
                    source_id,
                    mics.iter()
                        .enumerate()
                        .map(|(channel, (id, position))| {
                            if *source_id == "src-2" && matches!(channel, 4 | 5) {
                                return (*id, None);
                            }
                            let mut distance = distance_meter(*position, *truth);
                            if *source_id == "src-2" && channel == 3 {
                                distance += 0.15;
                            }
                            (*id, Some((distance, 0.001)))
                        })
                        .collect(),
                )
            })
            .collect();
        (plan, ranges)
    }

    #[test]
    fn joint_session_calibrates_unsurveyed_sources_and_mics() {
        // Both sides partially unsurveyed: the joint solve calibrates
        // all four unknowns coherently; surveyed sources keep forward
        // audits and surveyed microphones stay anchors.
        let mut harness = harness_rich(HarnessMode::Loopback);
        let source_truth = harness.golden.sources_m.clone();
        for source in &mut harness.report.plan.sources {
            source.position_m = source_truth.get(&source.id).copied();
            source.position_uncertainty_mm = Some(0.1);
        }
        for mic in harness.report.plan.microphones.iter_mut().skip(4) {
            mic.position_m = None;
            mic.position_uncertainty_mm = None;
        }
        let unsurveyed_sources: Vec<String> = harness
            .report
            .plan
            .sources
            .iter()
            .skip(3)
            .take(2)
            .map(|source| source.id.clone())
            .collect();
        assert_eq!(unsurveyed_sources.len(), 2);
        for source in &mut harness.report.plan.sources {
            if unsurveyed_sources.contains(&source.id) {
                source.position_m = None;
                source.position_uncertainty_mm = None;
            }
        }
        let localized = localize_session(
            &harness.report,
            harness.raw_root.path(),
            harness.output.path(),
            &harness.fits,
        );
        assert_eq!(localized.mode, LocalizationMode::LoopbackReferenced);
        let calibration = localized
            .joint
            .as_ref()
            .expect("joint calibration publishes");
        assert_eq!(calibration.sources_solved, 2);
        assert_eq!(calibration.microphones_solved, 2);
        assert_eq!(calibration.anchors, 7);
        assert!(!calibration.outlier_absorb_risk);
        for position in &localized.sources {
            let truth = harness.golden.sources_m[&position.source_id];
            let fit = position.position_m.expect("joint source solves");
            let error = distance_meter(fit, truth);
            assert!(
                error < 0.02,
                "{}: joint source error {error} m",
                position.source_id
            );
        }
        for microphone in &localized.microphones {
            if microphone.microphone_id == "mic-4" || microphone.microphone_id == "mic-5" {
                let truth = harness.golden.microphones_m[&microphone.microphone_id];
                let fit = microphone.position_m.expect("joint mic solves");
                let error = distance_meter(fit, truth);
                assert!(
                    error < 0.02,
                    "{}: joint mic error {error} m",
                    microphone.microphone_id
                );
            } else {
                assert_eq!(microphone.position_m, None);
                assert_eq!(microphone.reason.as_deref(), Some("surveyed_anchor"));
            }
        }
        let lines = localization_summary_lines(&localized);
        assert!(
            lines.iter().any(|line| line
                .starts_with("localization joint: 2 sources + 2 microphones from 7 anchors")),
            "joint line missing: {lines:?}"
        );
    }

    #[test]
    fn joint_q10_falls_back_to_direct_solves() {
        // Two anchors only: counting refuses the joint solve, direct
        // reasons stand, and a session note says the joint solve was
        // not attempted.
        let mut harness = harness_rich(HarnessMode::Loopback);
        for source in harness.report.plan.sources.iter_mut().skip(1) {
            source.position_m = None;
            source.position_uncertainty_mm = None;
        }
        {
            let first = harness.report.plan.sources[0].id.clone();
            let truth = harness.golden.sources_m[&first];
            harness.report.plan.sources[0].position_m = Some(truth);
            harness.report.plan.sources[0].position_uncertainty_mm = Some(0.1);
        }
        for mic in harness.report.plan.microphones.iter_mut().skip(1) {
            mic.position_m = None;
            mic.position_uncertainty_mm = None;
        }
        let localized = localize_session(
            &harness.report,
            harness.raw_root.path(),
            harness.output.path(),
            &harness.fits,
        );
        assert_eq!(localized.joint, None);
        assert!(
            localized.issues.iter().any(|issue| issue
                .contains("joint self-calibration not attempted")
                && issue.contains("direct solves stand")),
            "Q10 note missing: {:?}",
            localized.issues
        );
        for position in &localized.sources[1..] {
            assert_eq!(
                position.reason.as_deref(),
                Some("insufficient_ranges"),
                "direct reason stands for {}",
                position.source_id
            );
        }
        for microphone in &localized.microphones[1..] {
            assert_eq!(
                microphone.reason.as_deref(),
                Some("insufficient_ranges"),
                "direct reason stands for {}",
                microphone.microphone_id
            );
        }
    }

    #[test]
    fn joint_q10b_refusal_marks_solve_failed() {
        // Session counting passes but the solver depletes (src-2 loses
        // its wild fourth link to exclusion): every unsurveyed point is
        // refused with solve_failed while surveyed audits publish.
        let (plan, ranges) = joint_q10b_fixture();
        let attempt = try_joint_positions(&plan, &ranges, 343.0, 48_000).expect("counting passes");
        assert_eq!(attempt.calibration, None);
        for position in &attempt.sources {
            if position.source_id == "src-0" || position.source_id == "src-1" {
                assert!(position.position_m.is_some(), "surveyed audit publishes");
                assert!(position.declared_error_m.is_some());
            } else {
                assert_eq!(position.position_m, None);
                assert_eq!(position.reason.as_deref(), Some("solve_failed"));
            }
        }
        for microphone in &attempt.microphones {
            if microphone.microphone_id == "mic-3" || microphone.microphone_id == "mic-5" {
                assert_eq!(microphone.position_m, None);
                assert_eq!(microphone.reason.as_deref(), Some("solve_failed"));
            } else {
                assert_eq!(microphone.reason.as_deref(), Some("surveyed_anchor"));
            }
        }
        assert!(
            attempt
                .notes
                .iter()
                .any(|note| note.contains("joint self-calibration failed (solve_failed)")),
            "Q10b note missing: {:?}",
            attempt.notes
        );
    }

    #[test]
    fn bench_missing_file_falls_back_uncalibrated() {
        let mut harness = harness(HarnessMode::Loopback);
        harness.report.plan.localization_calibration_file = Some("missing.json".into());
        let localized = localize_session(
            &harness.report,
            harness.raw_root.path(),
            harness.output.path(),
            &harness.fits,
        );
        assert_eq!(localized.mic_bias_correction_m, 0.0);
        assert!(
            localized
                .issues
                .iter()
                .any(|issue| issue.contains("bench calibration unusable")),
            "missing bench file must warn: {:?}",
            localized.issues
        );
        // Ranges still solve uncalibrated.
        assert!(
            localized
                .ranges
                .iter()
                .flat_map(|row| row.cells.iter())
                .all(|cell| cell.distance_m.is_some())
        );
    }

    #[test]
    fn positions_publish_loo_diagnostics_with_spare_mics() {
        let microphones = vec![
            ("mic-0", [0.0, 0.0, 0.0]),
            ("mic-1", [1.0, 0.0, 0.0]),
            ("mic-2", [0.0, 1.0, 0.0]),
            ("mic-3", [0.0, 0.0, 1.0]),
            ("mic-4", [1.0, 1.0, 1.0]),
            ("mic-5", [1.0, 1.0, 0.0]),
        ];
        let truth = [2.3, -1.1, 0.7];
        let row = position_row(
            "left",
            microphones
                .iter()
                .map(|(id, position)| (*id, Some((distance_meter(*position, truth), 0.001))))
                .collect(),
        );
        let plan = position_plan(&microphones, Some(truth));
        let (sources, _) = solve_source_positions(&plan, &[row], 343.0, 48_000);
        let position = &sources[0];
        assert!(position.position_m.is_some());
        let residual = position.loo_max_residual_m.unwrap_or(f64::NAN);
        let shift = position.loo_max_shift_m.unwrap_or(f64::NAN);
        assert!(residual < 1.0e-6, "residual {residual}");
        assert!(shift < 1.0e-6, "shift {shift}");
        let worst = position.loo_worst_microphone.as_deref().unwrap_or("none");
        assert!(
            microphones.iter().any(|(id, _)| *id == worst),
            "worst mic {worst} must be a plan microphone"
        );
    }

    #[test]
    fn hf_detector_separates_full_and_low_content() {
        let rate = 48_000.0;
        let len = 12_000_usize;
        let sweep: Vec<f32> = (0..len)
            .map(|n| {
                let t = n as f64 / rate;
                (2.0 * std::f64::consts::PI * (20.0 * t + 7980.0 * t * t / 0.5)).sin() as f32
            })
            .collect();
        let full = hf_energy_fraction(&sweep, SUB_HF_CUTOFF_HZ, rate).expect("full sweep measures");
        assert!(full > 0.5, "full sweep HF fraction {full}");
        // Pure low-frequency tones stand in for a subwoofer recording:
        // filtering a chirp would leave edge transients that misstate
        // what a naturally band-limited recording looks like.
        let low_signal: Vec<f32> = (0..len)
            .map(|n| {
                let t = n as f64 / rate;
                ((2.0 * std::f64::consts::PI * 60.0 * t).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * 150.0 * t).sin()
                    + 0.3 * (2.0 * std::f64::consts::PI * 230.0 * t).sin()) as f32
            })
            .collect();
        let low =
            hf_energy_fraction(&low_signal, SUB_HF_CUTOFF_HZ, rate).expect("low signal measures");
        assert!(low < 0.01, "low HF fraction {low}");
        assert_eq!(hf_energy_fraction(&[], SUB_HF_CUTOFF_HZ, rate), None);
        assert_eq!(
            hf_energy_fraction(&vec![0.0; 128], SUB_HF_CUTOFF_HZ, rate),
            None
        );
    }

    #[test]
    fn sub_band_supports_only_lf_sweeps() {
        assert_eq!(sub_band(20.0, 120.0), Some((20.0, 120.0)));
        assert_eq!(sub_band(20.0, 300.0), Some((20.0, 300.0)));
        assert_eq!(sub_band(50.0, 20_000.0), Some((50.0, 300.0)));
        assert_eq!(sub_band(20.0, 60.0), None);
        assert_eq!(sub_band(2000.0, 8000.0), None);
        assert_eq!(sub_band(f64::NAN, 300.0), None);
    }

    #[test]
    fn detrend_removes_dc_and_drift() {
        let signal: Vec<f32> = (0..1000)
            .map(|n| 2.0 + 0.001 * n as f32 + (n as f32 * 0.1).sin() * 0.01)
            .collect();
        let clean = detrend_linear(&signal);
        assert_eq!(clean.len(), signal.len());
        let mean: f64 =
            clean.iter().map(|sample| f64::from(*sample)).sum::<f64>() / clean.len() as f64;
        assert!(mean.abs() < 1.0e-3, "residual mean {mean}");
        assert_eq!(detrend_linear(&[]).len(), 0);
        assert_eq!(detrend_linear(&[5.0]), vec![0.0]);
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
        assert!(
            lines
                .iter()
                .any(|line| line.contains("2/4 surveyed microphones"))
        );
        plan.microphones[0].position_m = None;
        plan.microphones[0].position_uncertainty_mm = None;
        let lines = localization_readiness(&plan);
        assert!(
            lines
                .iter()
                .any(|line| line.contains("1/2 microphones unsurveyed"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("1/4 surveyed microphones"))
        );
        plan.microphones[0].position_m = Some([0.0, 0.0, 1.0]);
        plan.microphones[0].position_uncertainty_mm = Some(0.1);
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
            microphone.position_m = Some(*position);
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
                    loo_max_residual_m: Some(0.004),
                    loo_worst_microphone: Some("mic-2".into()),
                    loo_max_shift_m: Some(0.01),
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
                    loo_max_residual_m: None,
                    loo_worst_microphone: None,
                    loo_max_shift_m: None,
                    reason: Some("solve_failed".into()),
                },
            ],
            issues: Vec::new(),
            microphones: Vec::new(),
            joint: None,
        };
        let lines = localization_summary_lines(&report);
        assert_eq!(lines.len(), 5);
        assert!(lines[2].contains("position: [2.300, -1.100, 0.700]"));
        assert!(lines[2].contains("rms 3.0 mm"));
        assert!(lines[2].contains("mirror-ambiguous"));
        assert!(lines[4].contains("refused (solve_failed)"));
    }

    #[test]
    fn summary_lines_show_microphones_only_past_survey() {
        let anchor = |id: &str| MicPosition {
            microphone_id: id.into(),
            position_m: None,
            sigma_m: None,
            rms_residual_m: None,
            gdop: None,
            mirror_ambiguous: false,
            declared_error_m: None,
            loo_max_residual_m: None,
            loo_worst_source: None,
            loo_max_shift_m: None,
            reason: Some("surveyed_anchor".into()),
        };
        let skeleton = |microphones: Vec<MicPosition>| LocalizationReport {
            mode: LocalizationMode::LoopbackReferenced,
            sound_speed_m_s: 343.0,
            sound_speed_uncertainty_m_s: 0.5,
            sound_speed_source: "plan".into(),
            pooled_skew_ppm: None,
            mic_bias_correction_m: 0.0,
            ranges: Vec::new(),
            sources: Vec::new(),
            microphones,
            joint: None,
            issues: Vec::new(),
        };
        // All surveyed: no microphone lines (Phase 3 output unchanged).
        let lines = localization_summary_lines(&skeleton(vec![anchor("mic-0"), anchor("mic-1")]));
        assert_eq!(lines.len(), 1);
        // One unsurveyed microphone: every mic gets a line.
        let mut estimated = anchor("mic-1");
        estimated.position_m = Some([0.1, 0.2, 1.0]);
        estimated.sigma_m = Some([0.01, 0.01, 0.02]);
        estimated.rms_residual_m = Some(0.002);
        estimated.gdop = Some(1.8);
        estimated.reason = None;
        let lines = localization_summary_lines(&skeleton(vec![anchor("mic-0"), estimated]));
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("mic mic-0 position: refused (surveyed_anchor)"));
        assert!(lines[2].contains("mic mic-1 position: [0.100, 0.200, 1.000]"));
        assert!(lines[2].contains("rms 2.0 mm"));
    }
}
