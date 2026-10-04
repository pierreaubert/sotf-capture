//! Canonical RoomEQ recording export with ordered per-microphone clock evidence.

use super::clock::CaptureClockBasis;
use super::clock::io::ClockProcessedManifest;
use super::{CalibrationOrientation, CaptureGeometry};
use autoeq::capture_provenance::{CaptureCorrection, CaptureProvenance, CaptureTakeProvenance};
use autoeq::read::{MeasurementMultiple, MeasurementRef};
use autoeq::roomeq::{MeasurementSource, RecordingConfiguration, RoomConfig, SpeakerConfig};
use autoeq::{MeasurementProvenance, ProvenanceCaptureKind};
use std::collections::{HashMap, HashSet};

fn measurement_quality_passed(analysis: &super::analysis::CaptureAnalysisReport) -> bool {
    analysis.magnitude_file.is_some()
        && analysis.clipped_samples == 0
        && analysis
            .broadband_snr_db
            .is_some_and(|value| value.is_finite() && value >= 30.0)
        && !analysis.frequency_snr.is_empty()
        && analysis.frequency_snr.iter().all(|band| {
            band.snr_db
                .is_some_and(|value| value.is_finite() && value >= 30.0)
        })
}

/// Build the canonical configuration from a complete set of analyzed capture takes.
///
/// Sources with complete accepted shared-reference artifacts export measured
/// phase. Other sources export magnitude. Clock bounds remain mandatory for
/// frequency-specific coherent eligibility. Calibration is already applied.
///
/// # Errors
/// Rejects incomplete or duplicated takes, missing calibration, and nonlocal artifacts.
pub fn recording_configuration(report: &ClockProcessedManifest) -> Result<RoomConfig, String> {
    report
        .plan
        .clone()
        .validate()
        .map_err(|error| error.to_string())?;
    let selected = selected_takes(report)?;
    let mut speakers = HashMap::new();
    for source in &report.plan.sources {
        let source_takes = report
            .plan
            .microphones
            .iter()
            .map(|mic| {
                selected
                    .get(&(source.id.as_str(), mic.id.as_str()))
                    .copied()
                    .ok_or_else(|| format!("{}/{}: selected take is missing", source.id, mic.id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let phase_complete = source_takes.iter().all(|take| {
            take.analysis.as_ref().is_some_and(|analysis| {
                analysis.common_reference.is_some()
                    && analysis.issues.is_empty()
                    && analysis.clipped_samples == 0
                    && analysis
                        .broadband_snr_db
                        .is_some_and(|value| value.is_finite() && value >= 30.0)
                    && !analysis.frequency_snr.is_empty()
                    && analysis.frequency_snr.iter().all(|band| {
                        band.snr_db
                            .is_some_and(|value| value.is_finite() && value >= 30.0)
                    })
            })
        });
        let mut measurements = Vec::new();
        let mut takes = Vec::new();
        for (mic, take) in report.plan.microphones.iter().zip(source_takes) {
            let file = take
                .analysis
                .as_ref()
                .and_then(|analysis| {
                    if phase_complete {
                        analysis
                            .common_reference
                            .as_ref()
                            .map(|artifacts| &artifacts.response_file)
                    } else {
                        analysis.magnitude_file.as_ref()
                    }
                })
                .ok_or_else(|| {
                    format!(
                        "{} / {}: calibrated magnitude is unavailable",
                        source.id, mic.id
                    )
                })?;
            if !autoeq::capture_handoff::portable_capture_filename(file) {
                return Err("measurement artifact must be a local filename".into());
            }
            measurements.push(MeasurementRef::Named {
                path: file.into(),
                name: Some(mic.id.clone()),
            });
            takes.push(take_provenance(report, take)?);
        }
        let timing_reference_id = takes
            .first()
            .and_then(|take| take.timing_reference_id.clone());
        let provenance = MeasurementProvenance {
            capture_kind: if phase_complete {
                ProvenanceCaptureKind::StationaryIr
            } else {
                ProvenanceCaptureKind::SpatialMagnitude
            },
            timing_reference_id,
            capture: Some(CaptureProvenance {
                reflection_report: report
                    .reflection_reports
                    .iter()
                    .find(|reflection| reflection.source_id == source.id)
                    .cloned(),
                geometry: match report.plan.geometry {
                    CaptureGeometry::Spread => autoeq::capture_provenance::CaptureGeometry::Spread,
                    CaptureGeometry::Compact => {
                        autoeq::capture_provenance::CaptureGeometry::Compact
                    }
                },
                takes,
            }),
            ..Default::default()
        };
        speakers.insert(
            source.id.clone(),
            SpeakerConfig::Single(MeasurementSource::Multiple(MeasurementMultiple {
                measurements,
                speaker_name: None,
                provenance,
            })),
        );
    }
    Ok(RoomConfig {
        speakers,
        recording_config: Some(RecordingConfiguration {
            capture_handoff_file: Some(autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME.into()),
            recording_sample_rate: Some(report.plan.sample_rate_hz),
            recording_channels: Some(report.plan.microphones.len()),
            signal_type: Some("Sweep".into()),
            signal_duration_secs: Some(report.plan.sweep.duration_secs as f32),
            signal_level_db: Some((20.0 * report.plan.sweep.amplitude.log10()) as f32),
            sweep_start_freq: Some(report.plan.sweep.start_hz as f32),
            sweep_end_freq: Some(report.plan.sweep.end_hz as f32),
            num_sweeps: Some(1),
            setup_description: Some("Calibrated multi-microphone capture. Phase requires accepted shared-reference artifacts and frequency-specific clock eligibility. See capture-clock.json for raw artifacts, timing bandwidth and review issues.".into()),
            ..Default::default()
        }),
        ..Default::default()
    })
}

fn selected_takes<'a>(
    report: &'a ClockProcessedManifest,
) -> Result<HashMap<(&'a str, &'a str), &'a super::clock::io::ClockProcessedTake>, String> {
    let selected_ids = if let Some(selected_ids) = &report.selected_take_ids {
        selected_ids
    } else if report.plan.repeat_count == 1
        && matches!(
            report.raw_status,
            None | Some(super::record::RawCaptureStatus::RawComplete)
        )
    {
        // Backward-compatible single-repeat projection. Older clock reports do
        // not carry raw_status or a selection marker; select_matrix still
        // requires exactly one completed take for every planned pair. Explicit
        // Cancelled/Failed/Capturing parents need a selection marker.
        return select_matrix(report, None);
    } else {
        return Err("partial or repeated captures require explicit selected take IDs".into());
    };
    select_matrix(report, Some(selected_ids))
}

fn select_matrix<'a>(
    report: &'a ClockProcessedManifest,
    selected_ids: Option<&[String]>,
) -> Result<HashMap<(&'a str, &'a str), &'a super::clock::io::ClockProcessedTake>, String> {
    let mut selected = HashMap::new();
    let mut ids = HashSet::new();
    if let Some(selected_ids) = selected_ids {
        for id in selected_ids {
            if !ids.insert(id.as_str()) {
                return Err(format!("selected take ID is duplicated: {id}"));
            }
            let take = report
                .takes
                .iter()
                .find(|take| take.raw.take_id == *id)
                .ok_or_else(|| format!("selected take ID is unknown: {id}"))?;
            if take.raw.repeat_index >= report.plan.repeat_count
                || take.raw.take_id.trim().is_empty()
            {
                return Err(format!(
                    "selected take ID is outside the declared domain: {id}"
                ));
            }
            let key = (take.raw.source_id.as_str(), take.raw.microphone_id.as_str());
            if selected.insert(key, take).is_some() {
                return Err(format!(
                    "selection repeats source/microphone pair {}/{}",
                    key.0, key.1
                ));
            }
        }
    } else {
        if report.plan.repeat_count != 1 {
            return Err("repeated captures require explicit selected take IDs".into());
        }
        for take in &report.takes {
            if take.raw.repeat_index != 0 {
                return Err("single-repeat capture contains an out-of-domain repeat index".into());
            }
            let key = (take.raw.source_id.as_str(), take.raw.microphone_id.as_str());
            if selected.insert(key, take).is_some() {
                return Err(format!(
                    "capture repeats source/microphone pair {}/{}",
                    key.0, key.1
                ));
            }
        }
    }
    let expected = report
        .plan
        .sources
        .len()
        .checked_mul(report.plan.microphones.len())
        .ok_or("selected capture matrix size overflow")?;
    if selected.len() != expected {
        return Err(
            "selection must contain one completed take per planned source/microphone pair".into(),
        );
    }
    for source in &report.plan.sources {
        for mic in &report.plan.microphones {
            if !selected.contains_key(&(source.id.as_str(), mic.id.as_str())) {
                return Err(format!(
                    "selected take is missing for {}/{}",
                    source.id, mic.id
                ));
            }
        }
    }
    Ok(selected)
}

pub(crate) fn take_provenance(
    report: &ClockProcessedManifest,
    take: &super::clock::io::ClockProcessedTake,
) -> Result<CaptureTakeProvenance, String> {
    let mic = report
        .plan
        .microphones
        .iter()
        .find(|mic| mic.id == take.raw.microphone_id)
        .ok_or("take microphone is outside the declared plan")?;
    let calibration = report
        .calibrations
        .iter()
        .find(|entry| entry.microphone_id == mic.id)
        .ok_or_else(|| format!("{}: calibration identity is unavailable", mic.id))?;
    Ok(CaptureTakeProvenance {
        microphone_id: mic.id.clone(),
        device_id: take.raw.device_id.clone(),
        offset_samples: take.clock.offset_samples,
        skew_ppm: take.clock.skew_ppm,
        residual_uncertainty_us: take.clock.residual_uncertainty_us,
        correction_applied: match take.clock.correction_applied.as_str() {
            "resampled" => CaptureCorrection::Resampled,
            "none" => CaptureCorrection::None,
            _ => return Err("unknown capture correction mode".into()),
        },
        timing_reference_id: take.clock.reference_id.clone(),
        calibration_id: calibration.sha256.clone(),
        gain_db: mic.gain_db,
        calibration_orientation: match mic.calibration_orientation {
            CalibrationOrientation::OnAxis => "on_axis",
            CalibrationOrientation::NinetyDegrees => "ninety_degrees",
        }
        .into(),
        position_m: mic.position_m,
        position_uncertainty_mm: mic.position_uncertainty_mm,
        preserves_acoustic_delay: take.clock.basis == CaptureClockBasis::FixedAcousticReference,
        quality_passed: take
            .analysis
            .as_ref()
            .is_some_and(measurement_quality_passed),
    })
}
