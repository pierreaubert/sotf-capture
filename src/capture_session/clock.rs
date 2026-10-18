//! Offline clock correction with explicit acoustic-reference provenance.
//!
//! An acoustic chirp includes propagation delay. Only a surveyed fixed emitter
//! permits separating that delay from device-clock offset. Without that evidence
//! resampling can align arrivals for magnitude analysis, but coherent use remains
//! disabled. Bounds assume linear device drift and isolated correctly identified
//! timing chirps; they do not certify arbitrary multipath or nonlinear drift.

use super::record::{RawCaptureManifest, RawCaptureTake};
use math_audio_dsp::capture_resample::{RESAMPLE_PHASES, resample_to_common_clock};
use math_audio_dsp::capture_tdoa::{
    ClockSkew, MAX_PLAUSIBLE_SKEW_PPM, TdoaConfig, TdoaEstimate, estimate_chirp_tdoa,
    estimate_clock_skew, post_correction_uncertainty_us,
};
use serde::{Deserialize, Serialize};

pub mod io;

/// Maximum per-channel timing uncertainty for the clock eligibility gate.
///
/// Fifty microseconds corresponds to nine degrees at 500 Hz. Higher-frequency
/// consumers must impose their own phase/aperture constraints in addition.
pub const COHERENT_TIMING_LIMIT_US: f64 = 50.0;

/// How a processed take's sample origin was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureClockBasis {
    /// Original samples; timing markers did not establish a usable affine fit.
    Uncorrected,
    /// Start-marker arrival was aligned, including unknown acoustic propagation.
    ArrivalAligned,
    /// Known acoustic travel time was retained on the fixed reference clock.
    FixedAcousticReference,
}

/// Clock fit and its uncertainty, persisted for each source/microphone take.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureClockProvenance {
    /// Physical input device ID from acquisition.
    pub device_id: String,
    /// Input index corresponding to stimulus sample zero, excluding known travel.
    pub offset_samples: Option<f64>,
    /// Input samples per stimulus interval, in parts per million.
    pub skew_ppm: Option<f64>,
    /// Conditional clock bound; absent means unknown, never zero.
    pub residual_uncertainty_us: Option<f64>,
    /// Chirp/model-only bound before adding surveyed geometry uncertainty.
    pub marker_uncertainty_us: Option<f64>,
    /// Start-marker peak confidence in dB.
    pub start_confidence_db: Option<f64>,
    /// End-marker peak confidence in dB.
    pub end_confidence_db: Option<f64>,
    /// `none` or `resampled`; raw recordings remain untouched.
    pub correction_applied: String,
    /// Clock origin evidence, distinct from correction having run.
    pub basis: CaptureClockBasis,
    /// Fixed output device/channel identity; coherent consumers must match it.
    pub reference_id: Option<String>,
    /// Maximum permitted skew used by this estimator.
    pub max_abs_skew_ppm: f64,
    /// Explicit reason coherent consumers must fall back to magnitude only.
    pub magnitude_only_reason: Option<String>,
}

/// A timing result with optional resampled samples.
#[derive(Debug)]
pub struct CorrectedCaptureClock {
    /// Correction and eligibility evidence, including refusal reasons.
    pub provenance: CaptureClockProvenance,
    /// Corrected samples; absent means retain the original magnitude-only take.
    pub samples: Option<Vec<f32>>,
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

// The correlation API reports signed circular lags. A late positive marker
// can lie past half the FFT length and be reported as negative. Acquisition
// searches require a complete marker inside the input window, so unwrap only
// when that positive interpretation fits; truly negative/truncated lags fail.
#[expect(
    clippy::cast_precision_loss,
    reason = "Validated capture windows contain at most 16 million samples."
)]
fn estimate_positive_chirp(
    reference: &[f32],
    recorded: &[f32],
    config: &TdoaConfig,
) -> TdoaEstimate {
    let mut estimate = estimate_chirp_tdoa(reference, recorded, config);
    if estimate.valid && estimate.offset_samples < 0.0 {
        let positive =
            estimate.offset_samples + (reference.len() + recorded.len()).next_power_of_two() as f64;
        if positive >= 0.0 && positive + reference.len() as f64 <= recorded.len() as f64 {
            estimate.offset_samples = positive;
        }
    }
    estimate
}

/// Affine clock fit for one take, independent of resampling.
///
/// The same fit that [`correct_capture_clock`] uses, exposed so later
/// stages (localization) can reuse the fitted lags and mapping without
/// re-running marker correlation on resampled audio or duplicating the
/// search-window logic. The fit is valid even when resampling later
/// refuses the mapping; consumers needing corrected audio must still
/// check `correction_applied` in the provenance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CaptureClockFit {
    /// Start-chirp lag referenced to the stimulus clock.
    pub first: TdoaEstimate,
    /// Clock mapping with the offset at stimulus sample zero.
    pub skew: ClockSkew,
}

/// Fit both markers without resampling the take.
///
/// Runs the identical estimation as [`correct_capture_clock`] and returns
/// the provenance plus the reusable fit. `None` carries the same refusal
/// provenance the resampling entry point returns, so behavior there is
/// unchanged: invalid or degraded markers yield a magnitude-only result,
/// never an error that discards the raw recording.
#[expect(
    clippy::too_many_lines,
    reason = "Keep the ordered acquisition and artifact-processing phases together so cleanup and provenance remain explicit."
)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "Capture lengths are bounded to 16 million samples; search windows are nonnegative and clamped to the recording."
)]
pub(crate) fn fit_capture_clock(
    manifest: &RawCaptureManifest,
    take: &RawCaptureTake,
    recorded: &[f32],
    reference: &[f32],
) -> (CaptureClockProvenance, Option<CaptureClockFit>) {
    let mut provenance = CaptureClockProvenance {
        device_id: take.device_id.clone(),
        offset_samples: None,
        skew_ppm: None,
        residual_uncertainty_us: None,
        marker_uncertainty_us: None,
        start_confidence_db: None,
        end_confidence_db: None,
        correction_applied: "none".into(),
        basis: CaptureClockBasis::Uncorrected,
        reference_id: None,
        max_abs_skew_ppm: MAX_PLAUSIBLE_SKEW_PPM,
        magnitude_only_reason: Some("timing markers did not establish a valid clock fit".into()),
    };
    let refused =
        |provenance: CaptureClockProvenance| -> (CaptureClockProvenance, Option<CaptureClockFit>) {
            (provenance, None)
        };
    let layout = &manifest.stimulus;
    let Some(spacing) = layout
        .end_chirp_offset
        .checked_sub(layout.start_chirp_offset)
        .filter(|n| *n > 0)
    else {
        return refused(provenance);
    };
    if reference.len() != layout.chirp_samples
        || reference.is_empty()
        || recorded.is_empty()
        || layout.sample_rate_hz <= 16_000
        || layout.total_samples == 0
        || layout.total_samples > 16_000_000
        || recorded.len() > 16_000_000
        || recorded.iter().chain(reference).any(|s| !s.is_finite())
    {
        return refused(provenance);
    }
    let config = TdoaConfig {
        sample_rate_hz: f64::from(layout.sample_rate_hz),
        ..TdoaConfig::default()
    };
    // Remove at least the marker separation from the tail so the identical end
    // marker cannot compete with the start marker. This tolerates arbitrary
    // device startup offsets rather than assuming host play() is synchronized.
    let minimum_spacing = (spacing as f64 * (1.0 - MAX_PLAUSIBLE_SKEW_PPM / 1e6)).floor() as usize;
    let first_end = recorded.len().saturating_sub(minimum_spacing);
    let first = estimate_positive_chirp(reference, &recorded[..first_end], &config);
    provenance.start_confidence_db = finite(first.confidence_db);
    if !first.valid
        || first.offset_samples < 0.0
        || first.offset_samples + reference.len() as f64 > first_end as f64
    {
        provenance.magnitude_only_reason =
            Some("start timing chirp is missing, ambiguous, or too weak".into());
        return refused(provenance);
    }
    let predicted_end = first.offset_samples + spacing as f64;
    let radius = spacing as f64 * MAX_PLAUSIBLE_SKEW_PPM / 1e6 + reference.len() as f64;
    let end_start = ((predicted_end - radius).max(0.0) as usize).min(recorded.len());
    let end_stop =
        ((predicted_end + radius + reference.len() as f64).ceil() as usize).min(recorded.len());
    let second = estimate_positive_chirp(reference, &recorded[end_start..end_stop], &config);
    provenance.end_confidence_db = finite(second.confidence_db);
    if !second.valid
        || second.offset_samples < 0.0
        || second.offset_samples + reference.len() as f64 > (end_stop - end_start) as f64
    {
        provenance.magnitude_only_reason =
            Some("end timing chirp is missing, ambiguous, or too weak".into());
        return refused(provenance);
    }
    let first_lag = TdoaEstimate {
        offset_samples: first.offset_samples - layout.start_chirp_offset as f64,
        ..first
    };
    let second_lag = TdoaEstimate {
        offset_samples: end_start as f64 + second.offset_samples - layout.end_chirp_offset as f64,
        ..second
    };
    let mut skew = estimate_clock_skew(&first_lag, &second_lag, spacing as f64);
    if !skew.valid {
        return refused(provenance);
    }
    provenance.skew_ppm = finite(skew.skew_ppm);
    let ratio = 1.0 + skew.skew_ppm / 1e6;
    // C2 returns lag at the first marker; C3 requires the sample-zero intercept.
    skew.offset_samples -= layout.start_chirp_offset as f64 * (ratio - 1.0);
    let model_bound = post_correction_uncertainty_us(
        &first_lag,
        &second_lag,
        &skew,
        spacing as f64,
        layout
            .total_samples
            .saturating_sub(layout.start_chirp_offset) as f64,
        &config,
    );
    // A fixed reference is not time-dilated before correlation. Conservatively
    // include the entire marker's skew-induced displacement and phase-table step.
    let marker_bound = model_bound
        + (reference.len() as f64 * (ratio - 1.0).abs() + 0.5 / RESAMPLE_PHASES as f64)
            / config.sample_rate_hz
            * 1e6;
    provenance.marker_uncertainty_us = finite(marker_bound);
    provenance.basis = CaptureClockBasis::ArrivalAligned;
    provenance.magnitude_only_reason = Some(
        "fixed timing-emitter geometry is unavailable; acoustic travel time is not a clock offset"
            .into(),
    );
    if let (Some(emitter), Some(microphone)) = (
        manifest.plan.timing_reference.as_ref(),
        manifest
            .plan
            .microphones
            .iter()
            .find(|mic| mic.id == take.microphone_id),
    ) && manifest.timing_reference_output_channel == Some(emitter.output_channel)
    {
        // An unsurveyed microphone cannot anchor the emitter geometry;
        // the take keeps its non-acoustic basis, as with non-finite
        // geometry today.
        let surveyed = match (microphone.position_m, microphone.position_uncertainty_mm) {
            (Some(position), Some(uncertainty)) => Some((position, uncertainty)),
            _ => None,
        };
        if let Some((mic_position, mic_uncertainty_mm)) = surveyed {
            let distance = mic_position
                .iter()
                .zip(emitter.position_m)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let speed = emitter.sound_speed_m_s;
            let minimum_speed = speed - emitter.sound_speed_uncertainty_m_s;
            if distance.is_finite() && minimum_speed > 0.0 && speed.is_finite() {
                skew.offset_samples -= distance / speed * config.sample_rate_hz * ratio;
                let geometry_bound = ((mic_uncertainty_mm + emitter.position_uncertainty_mm)
                    / 1000.0
                    / minimum_speed
                    + distance * emitter.sound_speed_uncertainty_m_s / (speed * minimum_speed))
                    * 1e6;
                provenance.residual_uncertainty_us = finite(marker_bound + geometry_bound);
                provenance.basis = CaptureClockBasis::FixedAcousticReference;
                provenance.reference_id = Some(format!(
                    "{}:{}",
                    take.output_device_id, emitter.output_channel
                ));
                provenance.magnitude_only_reason = match provenance.residual_uncertainty_us {
                    Some(bound) if (0.0..COHERENT_TIMING_LIMIT_US).contains(&bound) => None,
                    _ => Some(format!(
                        "timing uncertainty is unavailable or exceeds {COHERENT_TIMING_LIMIT_US} us"
                    )),
                };
            }
        }
    }
    provenance.offset_samples = finite(skew.offset_samples);
    let fit = CaptureClockFit {
        first: first_lag,
        skew,
    };
    (provenance, Some(fit))
}

/// Resample one take onto its fitted reference clock.
///
/// Applies the mapping from [`fit_capture_clock`] without re-running
/// estimation, so the processing pipeline can keep the fit for later
/// stages (localization) while producing the identical corrected take.
/// A refused resampling keeps the fit's provenance with an explicit
/// magnitude-only reason, exactly as before the fit was exposed.
pub(crate) fn apply_clock_fit(
    mut provenance: CaptureClockProvenance,
    fit: Option<CaptureClockFit>,
    recorded: &[f32],
    total_samples: usize,
) -> CorrectedCaptureClock {
    let Some(fit) = fit else {
        return CorrectedCaptureClock {
            provenance,
            samples: None,
        };
    };
    let Some(samples) = resample_to_common_clock(recorded, &fit.skew, total_samples) else {
        provenance.magnitude_only_reason =
            Some("common-clock resampling refused the fitted mapping".into());
        provenance.residual_uncertainty_us = None;
        return CorrectedCaptureClock {
            provenance,
            samples: None,
        };
    };
    provenance.correction_applied = "resampled".into();
    CorrectedCaptureClock {
        provenance,
        samples: Some(samples),
    }
}

/// Fit both markers and correct one take onto its declared reference clock.
///
/// Invalid or degraded markers return a magnitude-only result, not an error that
/// discards the raw recording. `manifest` and wave lengths are checked by the
/// folder-processing boundary before this deterministic operation.
#[must_use]
pub fn correct_capture_clock(
    manifest: &RawCaptureManifest,
    take: &RawCaptureTake,
    recorded: &[f32],
    reference: &[f32],
) -> CorrectedCaptureClock {
    let (provenance, fit) = fit_capture_clock(manifest, take, recorded, reference);
    apply_clock_fit(provenance, fit, recorded, manifest.stimulus.total_samples)
}

#[cfg(test)]
mod tests;
