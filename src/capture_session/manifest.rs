//! Canonical `RoomEQ` recording export with ordered per-microphone clock evidence.

// Rust guideline compliant 2026-02-21

use super::clock::CaptureClockBasis;
use super::clock::io::ClockProcessedManifest;
use super::{CalibrationOrientation, CaptureGeometry};
use autoeq::capture_provenance::{CaptureCorrection, CaptureProvenance, CaptureTakeProvenance};
use autoeq::read::{EstimatedPosition, EstimatedRangeCell, MeasurementMultiple, MeasurementRef};
use autoeq::roomeq::{
    BassAnchorChannelResultLegacy, BassAnchorResultsLegacy, MeasuredIrSource, MeasurementSource,
    ProbeChannelResultLegacy, ProbeResultsLegacy, RecordingConfiguration, RoomConfig,
    RoomDimensions, SpeakerConfig, validate_measured_ir_map,
};
use autoeq::{MeasurementProvenance, ProvenanceCaptureKind};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

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
/// Project one speaker's localization evidence into the handoff block (§8.4).
///
/// Emitted whenever the source has a localization entry: successes carry
/// the estimate plus its full range row, refusals carry the reason so
/// `RoomEQ` falls back to declared geometry with the cause attached.
/// Returns `None` when localization never ran for this source.
fn speaker_estimated_position(
    report: &ClockProcessedManifest,
    source_id: &str,
) -> Option<EstimatedPosition> {
    let localization = report.localization_report.as_ref()?;
    let position = localization
        .sources
        .iter()
        .find(|entry| entry.source_id == source_id)?;
    let ranges = localization
        .ranges
        .iter()
        .find(|row| row.source_id == source_id)
        .map(|row| {
            row.cells
                .iter()
                .map(|cell| EstimatedRangeCell {
                    microphone_id: cell.microphone_id.clone(),
                    distance_m: cell.distance_m,
                    sigma_m: cell.sigma_m,
                    reason: cell.reason.clone(),
                })
                .collect()
        })
        .unwrap_or_default();
    Some(EstimatedPosition {
        position_m: position.position_m,
        sigma_m: position.sigma_m,
        rms_residual_m: position.rms_residual_m,
        mirror_ambiguous: position.mirror_ambiguous,
        reason: position.reason.clone(),
        ranges,
    })
}

/// # Errors
/// Rejects incomplete or duplicated takes, missing calibration, and nonlocal artifacts.
#[expect(
    clippy::too_many_lines,
    reason = "Keep the ordered acquisition and artifact-processing phases together so cleanup and provenance remain explicit."
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "Validated capture levels and frequencies are intentionally exported through the f32 recording schema."
)]
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
                estimated_position: speaker_estimated_position(report, &source.id),
            })),
        );
    }
    Ok(RoomConfig {
        speakers,
        measured_impulse_responses: measured_impulse_responses(report)?,
        recording_config: Some(RecordingConfiguration {
            capture_handoff_file: Some(autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME.into()),
            recording_sample_rate: Some(report.plan.sample_rate_hz),
            recording_channels: Some(report.plan.microphones.len()),
            signal_type: Some("Sweep".into()),
            signal_duration_secs: Some(recording_signal_duration_secs(report)),
            bass_octave_duration_s: Some(recording_bass_octave_duration_s(&report.plan)),
            signal_level_db: Some((20.0 * report.plan.sweep.amplitude.log10()) as f32),
            sweep_start_freq: Some(report.plan.sweep.start_hz as f32),
            sweep_end_freq: Some(report.plan.sweep.end_hz as f32),
            num_sweeps: Some(1),
            room_dimensions: recording_room_dimensions(&report.plan),
            setup_description: Some(recording_setup_description(&report.plan)),
            probe_results: report
                .probe_pass
                .as_ref()
                .map(|pass| legacy_probe_results(&pass.results)),
            probe_wav_relative: report.probe_pass.as_ref().map(|pass| pass.wav_file.clone()),
            bass_anchor_results: report
                .bass_anchor_pass
                .as_ref()
                .map(|pass| legacy_bass_anchor_results(&pass.results)),
            bass_anchor_wav_relative: report
                .bass_anchor_pass
                .as_ref()
                .map(|pass| pass.wav_file.clone()),
            bass_probe_freq_hz: report
                .bass_anchor_pass
                .as_ref()
                .map(|pass| pass.results.bass_freq_hz),
            bass_probe_duration_s: report
                .bass_anchor_pass
                .as_ref()
                .map(|pass| pass.results.bass_duration_s),
            ..Default::default()
        }),
        ..Default::default()
    })
}

/// Default provenance note when the plan carries no setup description.
///
/// Kept distinct from user notes: plan text is passed through verbatim and
/// never merged with this fallback.
const DEFAULT_SETUP_DESCRIPTION: &str = "Calibrated multi-microphone capture. Phase requires accepted shared-reference artifacts and frequency-specific clock eligibility. See capture-clock.json for raw artifacts, timing bandwidth and review issues.";

fn recording_room_dimensions(plan: &super::CaptureSessionPlan) -> Option<RoomDimensions> {
    plan.room_dimensions.map(|room| RoomDimensions {
        length: room.length,
        width: room.width,
        height: room.height,
    })
}

fn recording_setup_description(plan: &super::CaptureSessionPlan) -> String {
    plan.setup_description
        .clone()
        .unwrap_or_else(|| DEFAULT_SETUP_DESCRIPTION.into())
}

/// Actual played sweep seconds from the journaled stimulus layout.
///
/// Bass weighting extends the sweep past its nominal duration; `RoomEQ`
/// must see the played span, not the nominal one.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "The recording schema stores duration in f32, matching the stimulus generator."
)]
fn recording_signal_duration_secs(report: &ClockProcessedManifest) -> f32 {
    report.stimulus.sweep_samples as f32 / f64::from(report.plan.sample_rate_hz) as f32
}

/// Bass dwell in seconds per octave below 100 Hz.
///
/// The plan value when bass weighting was played, else the effective dwell
/// of the plain log sweep (equal time per octave across the whole band).
#[expect(
    clippy::cast_possible_truncation,
    reason = "Validated sweep timing is intentionally converted to the f32 recording schema."
)]
fn recording_bass_octave_duration_s(plan: &super::CaptureSessionPlan) -> f32 {
    plan.sweep.bass_octave_duration_s.unwrap_or_else(|| {
        (plan.sweep.duration_secs / (plan.sweep.end_hz / plan.sweep.start_hz).log2()) as f32
    })
}

/// Translate runner probe results into the autoeq-local legacy mirror.
fn legacy_probe_results(results: &crate::signal_recorder::ProbeDelayResults) -> ProbeResultsLegacy {
    ProbeResultsLegacy {
        channels: results
            .channels
            .iter()
            .map(|channel| ProbeChannelResultLegacy {
                channel_name: channel.channel_name.clone(),
                channel_index: channel.channel_index,
                arrival_ms: channel.arrival_ms,
                gain_db: channel.gain_db,
                snr_db: channel.snr_db,
            })
            .collect(),
        sample_rate: results.sample_rate,
        alignment_delays_ms: results.alignment_delays_ms.clone(),
    }
}

/// Translate runner anchor results into the autoeq-local legacy mirror.
fn legacy_bass_anchor_results(
    results: &crate::signal_recorder::BassAnchorResults,
) -> BassAnchorResultsLegacy {
    BassAnchorResultsLegacy {
        channels: results
            .channels
            .iter()
            .map(|channel| BassAnchorChannelResultLegacy {
                channel_name: channel.channel_name.clone(),
                channel_index: channel.channel_index,
                bass_anchor_phase_deg: channel.bass_anchor_phase_deg,
                bass_anchor_magnitude: channel.bass_anchor_magnitude,
                bass_anchor_stability_deg: channel.bass_anchor_stability_deg,
                bass_anchor_loopback_phase_deg: channel.bass_anchor_loopback_phase_deg,
                bass_anchor_coherence: channel.bass_anchor_coherence,
            })
            .collect(),
        sample_rate: results.sample_rate,
        bass_freq_hz: results.bass_freq_hz,
        bass_duration_s: results.bass_duration_s,
    }
}

/// Measured room IR declarations, one per source, from one microphone.
///
/// `RoomEQ` admits a single IR per delivered channel, so the selected
/// microphone (explicit `--ir-microphone`, else the first plan microphone)
/// backs every source entry. Sources whose take lacks a deconvolved IR are
/// omitted; their report cells stay pending rather than borrowing audio.
fn measured_impulse_responses(
    report: &ClockProcessedManifest,
) -> Result<BTreeMap<String, MeasuredIrSource>, String> {
    let selected = selected_takes(report)?;
    let microphone = report
        .ir_microphone
        .as_deref()
        .or_else(|| {
            report
                .plan
                .microphones
                .first()
                .map(|microphone| microphone.id.as_str())
        })
        .ok_or("capture plan declares no microphones")?;
    if !report
        .plan
        .microphones
        .iter()
        .any(|candidate| candidate.id == microphone)
    {
        return Err(format!("unknown IR microphone: {microphone}"));
    }
    let mut declarations = BTreeMap::new();
    for source in &report.plan.sources {
        let take = selected
            .get(&(source.id.as_str(), microphone))
            .ok_or_else(|| format!("{}/{}: selected take is missing", source.id, microphone))?;
        let Some(csv) = take
            .analysis
            .as_ref()
            .and_then(|analysis| analysis.common_reference.as_ref())
            .and_then(|phase| phase.impulse_csv_file.clone())
        else {
            continue;
        };
        if !autoeq::capture_handoff::portable_capture_filename(&csv) {
            return Err("measured IR artifact must be a local filename".into());
        }
        declarations.insert(
            source.id.clone(),
            MeasuredIrSource {
                output_channel: None,
                driver: None,
                path: PathBuf::from(csv),
                sample_rate_hz: Some(f64::from(report.plan.sample_rate_hz)),
                timing_reference_id: take.clock.reference_id.clone(),
            },
        );
    }
    validate_measured_ir_map(&declarations)?;
    Ok(declarations)
}

pub(crate) fn selected_takes(
    report: &ClockProcessedManifest,
) -> Result<HashMap<(&str, &str), &super::clock::io::ClockProcessedTake>, String> {
    let Some(selected_ids) = &report.selected_take_ids else {
        if report.plan.repeat_count == 1
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
        }
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
    // The frozen autoeq handoff requires surveyed geometry per take; an
    // unsurveyed microphone refuses here and the manifest degrades to
    // pending_processing while the run (including localization) continues.
    let (Some(position_m), Some(position_uncertainty_mm)) =
        (mic.position_m, mic.position_uncertainty_mm)
    else {
        return Err(format!(
            "{}: RoomEQ handoff needs a surveyed microphone position",
            mic.id
        ));
    };
    Ok(CaptureTakeProvenance {
        seat_id: None,
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
        position_m,
        position_uncertainty_mm,
        preserves_acoustic_delay: take.clock.basis == CaptureClockBasis::FixedAcousticReference,
        quality_passed: take
            .analysis
            .as_ref()
            .is_some_and(measurement_quality_passed),
    })
}

#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "Synthetic fixtures use bounded sample counts and intentionally quantized audio; exact comparisons verify unchanged settings and samples."
)]
mod tests {
    use super::super::analysis::{CaptureAnalysisReport, CapturePhaseArtifacts};
    use super::super::clock::CaptureClockBasis;
    use super::super::clock::io::{ClockProcessedManifest, ClockProcessedTake};
    use super::super::protocol::CaptureStimulusLayout;
    use super::super::record::RawCaptureTake;
    use super::super::{CaptureSessionPlan, RoomDimensions};
    use super::{
        DEFAULT_SETUP_DESCRIPTION, measured_impulse_responses, recording_room_dimensions,
        recording_setup_description,
    };

    fn plan() -> CaptureSessionPlan {
        serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json")).unwrap()
    }

    fn fabricated_take(
        source_id: &str,
        microphone_id: &str,
        index: usize,
        impulse_csv: Option<&str>,
    ) -> ClockProcessedTake {
        ClockProcessedTake {
            raw: RawCaptureTake {
                take_id: format!("take-{source_id}-{microphone_id}"),
                repeat_index: 0,
                device_id: "usb".into(),
                output_device_id: "dac".into(),
                source_id: source_id.into(),
                microphone_id: microphone_id.into(),
                wav_file: format!("take-{index:03}.wav"),
                samples: 1024,
                input_sample_format: "F32".into(),
                output_sample_format: "F32".into(),
                peak_amplitude: 0.1,
                clipped_samples: 0,
            },
            audio_file: format!("take-{index:03}.wav"),
            clock: super::super::clock::CaptureClockProvenance {
                device_id: "usb".into(),
                offset_samples: Some(0.0),
                skew_ppm: Some(0.0),
                residual_uncertainty_us: Some(20.0),
                marker_uncertainty_us: Some(10.0),
                start_confidence_db: Some(30.0),
                end_confidence_db: Some(30.0),
                correction_applied: "resampled".into(),
                basis: CaptureClockBasis::FixedAcousticReference,
                reference_id: Some("dac:0".into()),
                max_abs_skew_ppm: 200.0,
                magnitude_only_reason: None,
            },
            analysis: Some(CaptureAnalysisReport {
                magnitude_file: Some(format!("take-{index:03}-magnitude.csv")),
                clipped_samples: 0,
                broadband_snr_db: None,
                frequency_snr: Vec::new(),
                common_reference: impulse_csv.map(|csv| CapturePhaseArtifacts {
                    response_file: format!("take-{index:03}-phase.csv"),
                    impulse_file: format!("take-{index:03}-ir.json"),
                    impulse_csv_file: Some(csv.into()),
                    sti: None,
                    coherence_file: None,
                    timing_limit_hz: 1250.0,
                    low_hz: 20.0,
                    high_hz: 20000.0,
                }),
                issues: Vec::new(),
                quality_findings: Vec::new(),
            }),
        }
    }

    fn fabricated_report(ir_microphone: Option<&str>) -> ClockProcessedManifest {
        let plan = plan();
        let takes = vec![
            fabricated_take("left", "seat-left", 0, Some("take-000-ir.csv")),
            fabricated_take("left", "seat-right", 1, None),
            fabricated_take("right", "seat-left", 2, Some("take-002-ir.csv")),
            fabricated_take("right", "seat-right", 3, None),
        ];
        ClockProcessedManifest {
            version: 1,
            plan,
            raw_status: None,
            selected_take_ids: None,
            ir_microphone: ir_microphone.map(str::to_owned),
            timing_reference_output_channel: Some(0),
            parent_inventory_file: None,
            stimulus: CaptureStimulusLayout {
                sample_rate_hz: 48000,
                start_chirp_offset: 0,
                end_chirp_offset: 0,
                chirp_samples: 0,
                sweep_offset: 0,
                sweep_samples: 5 * 48000,
                total_samples: 0,
            },
            calibrations: ["seat-left", "seat-right"]
                .into_iter()
                .enumerate()
                .map(
                    |(index, microphone)| super::super::record::CaptureCalibration {
                        microphone_id: microphone.into(),
                        file: format!("calibration-{index:02}.txt"),
                        sha256: "abc123".into(),
                    },
                )
                .collect(),
            takes,
            pending_processing: Vec::new(),
            probe_pass: None,
            bass_anchor_pass: None,
            loopback_files: Vec::new(),
            ctc_measurements: None,
            ctc_files: Vec::new(),
            recording_manifest: None,
            reflection_reports: Vec::new(),
            localization_report: None,
        }
    }

    #[test]
    fn measured_map_uses_first_microphone_by_default() {
        let map = measured_impulse_responses(&fabricated_report(None)).unwrap();
        assert_eq!(map.len(), 2);
        for (source, file) in [("left", "take-000-ir.csv"), ("right", "take-002-ir.csv")] {
            let entry = &map[source];
            assert_eq!(entry.path.to_str(), Some(file));
            assert_eq!(entry.sample_rate_hz, Some(48000.0));
            assert_eq!(entry.timing_reference_id.as_deref(), Some("dac:0"));
            assert!(entry.output_channel.is_none());
            assert!(entry.driver.is_none());
        }
    }

    #[test]
    fn emitted_estimated_position_loads_into_autoeq_gate() {
        // Phase 5 handshake: the recordings projection carries each
        // speaker's localization block, RoomEQ parses it, and the
        // delay-alignment gate accepts a tight estimate while naming
        // a refused one's fallback.
        use super::super::localization::{
            LocalizationMode, LocalizationReport, RangeCell, SourcePosition, SourceRanges,
        };
        use autoeq::roomeq::{MeasurementSource, RoomConfig, SpeakerConfig};
        let mut report = fabricated_report(None);
        let cell = |microphone_id: &str| RangeCell {
            microphone_id: microphone_id.into(),
            distance_m: Some(2.5),
            sigma_m: Some(0.002),
            arrival_samples: None,
            reference_arrival_samples: None,
            tdoa_samples_vs_first: None,
            tdoa_sigma_samples: None,
            band: None,
            confidence_db: None,
            reason: None,
        };
        let solved = SourcePosition {
            source_id: "left".into(),
            position_m: Some([2.3, -1.1, 0.7]),
            sigma_m: Some([0.004, 0.005, 0.006]),
            rms_residual_m: Some(0.003),
            gdop: None,
            mirror_ambiguous: false,
            declared_error_m: None,
            loo_max_residual_m: None,
            loo_worst_microphone: None,
            loo_max_shift_m: None,
            reason: None,
        };
        let refused = SourcePosition {
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
        };
        report.localization_report = Some(LocalizationReport {
            mode: LocalizationMode::LoopbackReferenced,
            sound_speed_m_s: 343.0,
            sound_speed_uncertainty_m_s: 0.5,
            sound_speed_source: "test".into(),
            pooled_skew_ppm: None,
            mic_bias_correction_m: 0.0,
            ranges: ["left", "right"]
                .map(|source_id| SourceRanges {
                    source_id: source_id.into(),
                    cells: ["seat-left", "seat-right"].map(cell).to_vec(),
                })
                .to_vec(),
            sources: vec![solved, refused],
            microphones: Vec::new(),
            joint: None,
            issues: Vec::new(),
        });
        let configuration = super::recording_configuration(&report).unwrap();
        let room: RoomConfig =
            serde_json::from_value(serde_json::to_value(&configuration).unwrap()).unwrap();
        let SpeakerConfig::Single(MeasurementSource::Multiple(left)) = &room.speakers["left"]
        else {
            panic!("left speaker keeps the capture projection shape");
        };
        let block = left
            .estimated_position
            .as_ref()
            .expect("tight estimate projects");
        assert_eq!(block.gated(), Ok([2.3, -1.1, 0.7]));
        assert_eq!(block.ranges.len(), 2);
        assert_eq!(block.ranges[0].microphone_id, "seat-left");
        assert_eq!(block.ranges[0].distance_m, Some(2.5));
        let SpeakerConfig::Single(MeasurementSource::Multiple(right)) = &room.speakers["right"]
        else {
            panic!("right speaker keeps the capture projection shape");
        };
        let block = right
            .estimated_position
            .as_ref()
            .expect("refusals carry the reason");
        assert_eq!(block.gated(), Err("refused_upstream"));
    }

    #[test]
    fn measured_map_omits_sources_without_a_deconvolved_ir() {
        let map = measured_impulse_responses(&fabricated_report(Some("seat-right"))).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn measured_map_refuses_unknown_microphones() {
        assert!(measured_impulse_responses(&fabricated_report(Some("seat-ghost"))).is_err());
    }

    #[test]
    fn configuration_projects_auxiliary_passes_verbatim() {
        use super::super::record::{CaptureBassAnchorPass, CaptureProbePass};
        use crate::signal_recorder::{
            BassAnchorChannelResult, BassAnchorResults, ProbeDelayChannelResult, ProbeDelayResults,
        };

        let mut report = fabricated_report(None);
        report.probe_pass = Some(CaptureProbePass {
            wav_file: "probe.wav".into(),
            microphone_id: "seat-left".into(),
            results: ProbeDelayResults {
                channels: vec![ProbeDelayChannelResult {
                    channel_name: "left".into(),
                    channel_index: 0,
                    arrival_ms: 1.5,
                    gain_db: -3.0,
                    snr_db: 40.0,
                }],
                sample_rate: 48000,
                alignment_delays_ms: vec![0.5],
            },
        });
        report.bass_anchor_pass = Some(CaptureBassAnchorPass {
            wav_file: "bass-anchor.wav".into(),
            microphone_id: "seat-left".into(),
            loopback_channel: Some(7),
            results: BassAnchorResults {
                channels: vec![BassAnchorChannelResult {
                    channel_name: "left".into(),
                    channel_index: 0,
                    bass_anchor_phase_deg: 12.0,
                    bass_anchor_magnitude: 0.5,
                    bass_anchor_stability_deg: 2.0,
                    bass_anchor_loopback_phase_deg: Some(3.0),
                    bass_anchor_coherence: Some(0.98),
                }],
                sample_rate: 48000,
                bass_freq_hz: 30.0,
                bass_duration_s: 2.0,
            },
        });
        let configuration = super::recording_configuration(&report).unwrap();
        let recording = configuration.recording_config.as_ref().unwrap();
        let probe = recording.probe_results.as_ref().unwrap();
        assert_eq!(probe.channels[0].arrival_ms, 1.5);
        assert_eq!(probe.alignment_delays_ms, vec![0.5]);
        assert_eq!(probe.sample_rate, 48000);
        assert_eq!(recording.probe_wav_relative.as_deref(), Some("probe.wav"));
        let anchor = recording.bass_anchor_results.as_ref().unwrap();
        assert_eq!(anchor.channels[0].bass_anchor_phase_deg, 12.0);
        assert_eq!(anchor.channels[0].bass_anchor_coherence, Some(0.98));
        assert_eq!(
            recording.bass_anchor_wav_relative.as_deref(),
            Some("bass-anchor.wav")
        );
        assert_eq!(recording.bass_probe_freq_hz, Some(30.0));
        assert_eq!(recording.bass_probe_duration_s, Some(2.0));

        let bare = super::recording_configuration(&fabricated_report(None)).unwrap();
        let recording = bare.recording_config.as_ref().unwrap();
        assert_eq!(recording.signal_duration_secs, Some(5.0));
        let effective = (5.0 / 1000f64.log2()) as f32;
        assert!((recording.bass_octave_duration_s.unwrap() - effective).abs() < 1e-6);
        assert!(recording.probe_results.is_none());
        assert!(recording.probe_wav_relative.is_none());
        assert!(recording.bass_anchor_results.is_none());
        assert!(recording.bass_anchor_wav_relative.is_none());
        assert!(recording.bass_probe_freq_hz.is_none());
        assert!(recording.bass_probe_duration_s.is_none());

        let mut weighted = fabricated_report(None);
        weighted.plan.sweep.bass_octave_duration_s = Some(3.0);
        let weighted = super::recording_configuration(&weighted).unwrap();
        assert_eq!(
            weighted
                .recording_config
                .as_ref()
                .unwrap()
                .bass_octave_duration_s,
            Some(3.0)
        );
    }

    #[test]
    fn room_projection_passes_dimensions_through_verbatim() {
        let mut value = plan();
        assert!(recording_room_dimensions(&value).is_none());
        value.room_dimensions = Some(RoomDimensions {
            length: 5.0,
            width: 4.0,
            height: 2.5,
        });
        let projected = recording_room_dimensions(&value).unwrap();
        assert_eq!(projected.length, 5.0);
        assert_eq!(projected.width, 4.0);
        assert_eq!(projected.height, 2.5);
    }

    #[test]
    fn setup_description_falls_back_without_merging() {
        let mut value = plan();
        assert_eq!(
            recording_setup_description(&value),
            DEFAULT_SETUP_DESCRIPTION
        );
        value.setup_description = Some("Treated room".into());
        assert_eq!(recording_setup_description(&value), "Treated room");
    }
}
