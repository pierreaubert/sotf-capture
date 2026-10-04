use super::*;
use crate::capture_session::protocol::prepare_capture_stimulus;
use crate::capture_session::record::RawCaptureStatus;
use crate::capture_session::{CaptureGeometry, CaptureSessionPlan, CaptureTimingReference};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Golden {
    sample_rate_hz: u32,
    offset_samples: [f64; 2],
    skew_ppm: [f64; 2],
    microphone_positions_m: [[f64; 3]; 2],
    timing_emitter_position_m: [f64; 3],
    sound_speed_m_s: f64,
}

fn fixture() -> (Golden, RawCaptureManifest, Vec<f32>, Vec<f32>) {
    let golden: Golden = serde_json::from_str(include_str!(
        "../../../tests/fixtures/capture-clock-golden.json"
    ))
    .unwrap();
    let mut plan: CaptureSessionPlan =
        serde_json::from_str(include_str!("../../../tests/fixtures/capture-session.json")).unwrap();
    plan.sample_rate_hz = golden.sample_rate_hz;
    plan.geometry = CaptureGeometry::Compact;
    plan.sweep.duration_secs = 0.05;
    for (mic, position) in plan
        .microphones
        .iter_mut()
        .zip(golden.microphone_positions_m)
    {
        mic.position_m = position;
        mic.position_uncertainty_mm = 0.1;
    }
    plan.timing_reference = Some(CaptureTimingReference {
        output_channel: 0,
        position_m: golden.timing_emitter_position_m,
        position_uncertainty_mm: 0.1,
        sound_speed_m_s: golden.sound_speed_m_s,
        sound_speed_uncertainty_m_s: 0.1,
    });
    let session = plan.validate().unwrap();
    let stimulus = prepare_capture_stimulus(&session).unwrap();
    let manifest = RawCaptureManifest {
        version: 1,
        plan: session.plan().clone(),
        stimulus: stimulus.layout,
        timing_reference_output_channel: Some(0),
        status: RawCaptureStatus::RawComplete,
        pending_processing: vec!["clock_correction".into()],
        calibrations: vec![],
        takes: vec![],
        error: None,
    };
    (golden, manifest, stimulus.samples, stimulus.timing_chirp)
}

// Independent first-order-hold sampling oracle. This intentionally does not use
// the production FIR resampler; the interpolation adds a small waveform error.
fn sample_clock(stimulus: &[f32], offset: f64, ppm: f64, propagation: f64) -> Vec<f32> {
    (0..stimulus.len() + 4096)
        .map(|n| {
            let time = (n as f64 - offset) / (1.0 + ppm / 1e6) - propagation;
            if time < 0.0 || time >= (stimulus.len() - 1) as f64 {
                return 0.0;
            }
            let index = time.floor() as usize;
            let frac = (time - index as f64) as f32;
            stimulus[index] * (1.0 - frac) + stimulus[index + 1] * frac
        })
        .collect()
}

fn take(manifest: &RawCaptureManifest, index: usize, samples: usize) -> RawCaptureTake {
    RawCaptureTake {
        take_id: crate::capture_session::record::stable_take_id(
            "left",
            &manifest.plan.microphones[index].id,
            0,
        ),
        repeat_index: 0,
        device_id: format!("usb-{index}"),
        output_device_id: "dac".into(),
        source_id: "left".into(),
        microphone_id: manifest.plan.microphones[index].id.clone(),
        wav_file: format!("mic-{index}.wav"),
        samples,
        input_sample_format: "F32".into(),
        output_sample_format: "F32".into(),
        peak_amplitude: 0.1,
        clipped_samples: 0,
    }
}

#[test]
fn golden_two_microphone_clock_fit_preserves_physical_arrival_differences() {
    let (golden, manifest, stimulus, chirp) = fixture();
    for index in 0..2 {
        let propagation = (1.0 - golden.microphone_positions_m[index][0]) / golden.sound_speed_m_s
            * f64::from(golden.sample_rate_hz);
        let input = sample_clock(
            &stimulus,
            golden.offset_samples[index],
            golden.skew_ppm[index],
            propagation,
        );
        let result = correct_capture_clock(
            &manifest,
            &take(&manifest, index, input.len()),
            &input,
            &chirp,
        );
        let clock = result.provenance;
        assert_eq!(
            clock.basis,
            CaptureClockBasis::FixedAcousticReference,
            "{clock:?}"
        );
        assert!(
            (clock.skew_ppm.unwrap() - golden.skew_ppm[index]).abs() < 2.0,
            "{clock:?}"
        );
        assert!(
            (clock.offset_samples.unwrap() - golden.offset_samples[index]).abs() < 1.0,
            "{clock:?}"
        );
        let bound = clock.residual_uncertainty_us.unwrap();
        assert!(bound.is_finite() && bound > 0.0);
        let corrected = result.samples.unwrap();
        let measured = estimate_chirp_tdoa(
            &chirp,
            &corrected[..manifest.stimulus.sweep_offset],
            &TdoaConfig::default(),
        );
        let error_samples =
            (measured.offset_samples - manifest.stimulus.start_chirp_offset as f64 - propagation)
                .abs();
        assert!(
            error_samples / f64::from(golden.sample_rate_hz) * 1e6 < bound,
            "error {error_samples}, bound {bound}"
        );
        assert!(
            measured.offset_samples - manifest.stimulus.start_chirp_offset as f64 > 100.0,
            "physical propagation must not be silently removed"
        );
    }
}

#[test]
fn missing_end_chirp_keeps_the_take_magnitude_only() {
    let (golden, manifest, stimulus, chirp) = fixture();
    let mut input = sample_clock(&stimulus, golden.offset_samples[0], golden.skew_ppm[0], 0.0);
    let start =
        (manifest.stimulus.end_chirp_offset as f64 * 1.00008 + golden.offset_samples[0]) as usize;
    input[start.saturating_sub(32)..].fill(0.0);
    let result = correct_capture_clock(&manifest, &take(&manifest, 0, input.len()), &input, &chirp);
    assert!(result.samples.is_none());
    assert_eq!(result.provenance.correction_applied, "none");
    assert!(result.provenance.residual_uncertainty_us.is_none());
    assert!(
        result
            .provenance
            .magnitude_only_reason
            .unwrap()
            .contains("end timing chirp")
    );
}

#[test]
fn absent_survey_never_turns_arrival_alignment_into_coherent_evidence() {
    let (golden, mut manifest, stimulus, chirp) = fixture();
    manifest.plan.timing_reference = None;
    let input = sample_clock(
        &stimulus,
        golden.offset_samples[0],
        golden.skew_ppm[0],
        140.0,
    );
    let result = correct_capture_clock(&manifest, &take(&manifest, 0, input.len()), &input, &chirp);
    assert!(result.samples.is_some());
    assert_eq!(result.provenance.basis, CaptureClockBasis::ArrivalAligned);
    assert!(result.provenance.residual_uncertainty_us.is_none());
    assert!(result.provenance.magnitude_only_reason.is_some());
}

fn write_golden_directory() -> tempfile::TempDir {
    use crate::capture_session::record::CaptureCalibration;
    use crate::signal_recorder::write_wav_file;
    use sha2::{Digest, Sha256};
    let root = tempfile::tempdir().unwrap();
    let (golden, mut manifest, stimulus, chirp) = fixture();
    manifest.plan.sources.truncate(1);
    write_wav_file(
        &root.path().join("stimulus.wav"),
        &stimulus,
        golden.sample_rate_hz,
        1,
    )
    .unwrap();
    write_wav_file(
        &root.path().join("timing-chirp.wav"),
        &chirp,
        golden.sample_rate_hz,
        1,
    )
    .unwrap();
    for index in 0..2 {
        let propagation = (1.0 - golden.microphone_positions_m[index][0]) / golden.sound_speed_m_s
            * f64::from(golden.sample_rate_hz);
        let mut input = sample_clock(
            &stimulus,
            golden.offset_samples[index],
            golden.skew_ppm[index],
            propagation,
        );
        if index == 1 {
            let end_start = ((manifest.stimulus.end_chirp_offset as f64 + propagation)
                * (1.0 + golden.skew_ppm[index] / 1e6)
                + golden.offset_samples[index]) as usize;
            input[end_start.saturating_sub(32)..].fill(0.0);
        }
        let take = take(&manifest, index, input.len());
        write_wav_file(
            &root.path().join(&take.wav_file),
            &input,
            golden.sample_rate_hz,
            1,
        )
        .unwrap();
        manifest.takes.push(take);
        let bytes = b"20 0\n1000 0\n20000 0\n";
        let file = format!("calibration-{index}.txt");
        std::fs::write(root.path().join(&file), bytes).unwrap();
        manifest.calibrations.push(CaptureCalibration {
            microphone_id: manifest.plan.microphones[index].id.clone(),
            file,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        });
    }
    std::fs::write(
        root.path().join("capture-raw.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    root
}

#[test]
fn saved_golden_session_round_trips_clock_provenance_and_magnitude_fallback() {
    let root = write_golden_directory();
    let output = root.path().join("processed");
    let report = io::process_capture_session(root.path(), &output).unwrap();
    assert_eq!(report.takes.len(), 2);
    assert_eq!(report.reflection_reports.len(), report.plan.sources.len());
    assert!(
        report
            .reflection_reports
            .iter()
            .all(|source| !source.issues.is_empty())
    );
    assert_eq!(report.takes[0].clock.correction_applied, "resampled");
    assert!(report.takes[0].clock.residual_uncertainty_us.is_some());
    assert_eq!(report.takes[1].clock.correction_applied, "none");
    assert!(report.takes[1].clock.residual_uncertainty_us.is_none());
    assert_eq!(
        report.recording_manifest.as_deref(),
        Some("recordings.json"),
        "{:?}",
        report.pending_processing
    );
    let handoff: autoeq::capture_handoff::CaptureHandoff = serde_json::from_slice(
        &std::fs::read(output.join(autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME)).unwrap(),
    )
    .unwrap();
    handoff.validate().unwrap();
    assert_eq!(
        handoff.completion,
        autoeq::capture_handoff::CaptureCompletion::Complete
    );
    assert_eq!(handoff.takes.len(), 2);
    for (index, take) in handoff.takes.iter().enumerate() {
        let retained = std::fs::read(output.join(&take.raw_audio_file)).unwrap();
        let original = std::fs::read(root.path().join(&report.takes[index].raw.wav_file)).unwrap();
        assert_eq!(retained, original);
        let identity = handoff
            .artifacts
            .iter()
            .find(|asset| asset.file == take.raw_audio_file)
            .unwrap();
        assert_eq!(identity.sha256, format!("{:x}", Sha256::digest(&retained)));
    }
    // Exercise the actual producer-to-consumer handoff.
    autoeq::roomeq::load_config(&output.join("recordings.json"), None).unwrap();
    let configuration: autoeq::roomeq::RoomConfig =
        serde_json::from_slice(&std::fs::read(output.join("recordings.json")).unwrap()).unwrap();
    let autoeq::roomeq::SpeakerConfig::Single(source) = &configuration.speakers["left"] else {
        panic!("expected captured source");
    };
    let autoeq::roomeq::MeasurementSource::Multiple(multiple) = source else {
        panic!("expected ordered microphone measurements");
    };
    let capture = multiple.provenance.capture.as_ref().unwrap();
    assert_eq!(capture.takes.len(), 2);
    assert_eq!(
        capture.reflection_report.as_ref().unwrap().source_id,
        "left"
    );
    assert!(
        !capture
            .reflection_report
            .as_ref()
            .unwrap()
            .issues
            .is_empty()
    );
    assert_eq!(
        capture.takes[0].microphone_id,
        report.plan.microphones[0].id
    );
    assert_eq!(
        capture.takes[0].calibration_id,
        report.calibrations[0].sha256
    );
    assert_eq!(
        capture.takes[1].correction_applied,
        autoeq::capture_provenance::CaptureCorrection::None
    );
    assert!(capture.takes[1].residual_uncertainty_us.is_none());
    assert!(source.provenance().timing_reference_id.is_none());
    assert_eq!(
        source.provenance().capture_kind,
        autoeq::ProvenanceCaptureKind::SpatialMagnitude
    );
    let mut incomplete = report.clone();
    incomplete.takes.pop();
    assert!(crate::capture_session::manifest::recording_configuration(&incomplete).is_err());
    let original = hound::WavReader::open(root.path().join(&report.takes[1].raw.wav_file))
        .unwrap()
        .into_samples::<f32>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let fallback = hound::WavReader::open(output.join(&report.takes[1].audio_file))
        .unwrap()
        .into_samples::<f32>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        original, fallback,
        "degraded timing must preserve usable raw magnitude samples"
    );
    let persisted: io::ClockProcessedManifest =
        serde_json::from_slice(&std::fs::read(output.join("capture-clock.json")).unwrap()).unwrap();
    assert_eq!(
        persisted.takes[0].clock.reference_id.as_deref(),
        Some("dac:0")
    );
    assert!(
        io::process_capture_session(root.path(), &output).is_err(),
        "never overwrite existing artifacts"
    );
}

#[test]
fn canonical_phase_requires_complete_quality_and_retains_frequency_gate() {
    use crate::capture_session::analysis::{CaptureFrequencySnr, CapturePhaseArtifacts};
    let root = write_golden_directory();
    let output = root.path().join("processed");
    let mut report = io::process_capture_session(root.path(), &output).unwrap();
    let clock = report.takes[0].clock.clone();
    for (index, take) in report.takes.iter_mut().enumerate() {
        take.clock = clock.clone();
        take.clock.device_id = take.raw.device_id.clone();
        take.clock.residual_uncertainty_us = Some(20.0);
        take.clock.magnitude_only_reason = None;
        let analysis = take.analysis.as_mut().unwrap();
        analysis.issues.clear();
        analysis.clipped_samples = 0;
        analysis.broadband_snr_db = Some(40.0);
        analysis.frequency_snr = vec![CaptureFrequencySnr {
            low_hz: 20.0,
            high_hz: 20000.0,
            snr_db: Some(40.0),
        }];
        analysis.common_reference = Some(CapturePhaseArtifacts {
            response_file: format!("take-{index:03}-phase.csv"),
            impulse_file: format!("take-{index:03}-ir.json"),
            timing_limit_hz: 1250.0,
            low_hz: 20.0,
            high_hz: 20000.0,
        });
    }
    let config = crate::capture_session::manifest::recording_configuration(&report).unwrap();
    let autoeq::roomeq::SpeakerConfig::Single(source) = &config.speakers["left"] else {
        panic!()
    };
    assert_eq!(
        source.provenance().capture_kind,
        autoeq::ProvenanceCaptureKind::StationaryIr
    );
    let provenance = source.provenance();
    let capture = provenance.capture.unwrap();
    assert!(capture.coherent_reference_at_frequency(2, 500.0).is_ok());
    assert!(capture.coherent_reference_at_frequency(2, 20000.0).is_err());
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("take-000-phase.csv"));
    report.takes[1].analysis.as_mut().unwrap().common_reference = None;
    let magnitude_only =
        crate::capture_session::manifest::recording_configuration(&report).unwrap();
    let autoeq::roomeq::SpeakerConfig::Single(source) = &magnitude_only.speakers["left"] else {
        panic!("captured source");
    };
    let provenance = source.provenance();
    assert_eq!(
        provenance.capture_kind,
        autoeq::ProvenanceCaptureKind::SpatialMagnitude
    );
    assert!(
        provenance
            .capture
            .unwrap()
            .takes
            .iter()
            .all(|take| take.quality_passed),
        "missing shared phase must not erase separately accepted magnitude quality"
    );
    report.takes[1].analysis.as_mut().unwrap().frequency_snr[0].snr_db = None;
    let fallback = crate::capture_session::manifest::recording_configuration(&report).unwrap();
    let autoeq::roomeq::SpeakerConfig::Single(source) = &fallback.speakers["left"] else {
        panic!()
    };
    assert_eq!(
        source.provenance().capture_kind,
        autoeq::ProvenanceCaptureKind::SpatialMagnitude
    );
    assert!(
        !serde_json::to_string(&fallback)
            .unwrap()
            .contains("phase.csv")
    );
}

#[test]
fn legacy_clock_report_without_raw_status_requires_the_complete_single_repeat_matrix() {
    let root = write_golden_directory();
    let output = root.path().join("legacy-processed");
    let report = io::process_capture_session(root.path(), &output).unwrap();
    let mut value = serde_json::to_value(report).unwrap();
    let report = value.as_object_mut().unwrap();
    report.remove("raw_status");
    report.remove("selected_take_ids");
    report.remove("parent_inventory_file");
    value["plan"]
        .as_object_mut()
        .unwrap()
        .remove("repeat_count");

    let legacy: io::ClockProcessedManifest = serde_json::from_value(value).unwrap();
    let configuration = crate::capture_session::manifest::recording_configuration(&legacy)
        .expect("legacy reports retain complete single-repeat compatibility");
    assert_eq!(configuration.speakers.len(), legacy.plan.sources.len());

    let mut incomplete = legacy.clone();
    incomplete.takes.pop();
    assert!(
        crate::capture_session::manifest::recording_configuration(&incomplete).is_err(),
        "legacy compatibility must still require every source/microphone pair"
    );

    for status in [RawCaptureStatus::Cancelled, RawCaptureStatus::Failed] {
        let mut explicitly_incomplete = legacy.clone();
        explicitly_incomplete.raw_status = Some(status);
        assert!(
            crate::capture_session::manifest::recording_configuration(&explicitly_incomplete)
                .is_err(),
            "explicit {status:?} status must not use the legacy projection"
        );
    }
}

#[test]
fn altered_calibration_snapshot_is_rejected_before_output_creation() {
    let root = write_golden_directory();
    std::fs::write(root.path().join("calibration-1.txt"), "changed").unwrap();
    let output = root.path().join("processed");
    let error = io::process_capture_session(root.path(), &output).unwrap_err();
    assert!(error.contains("has changed"), "{error}");
    assert!(!output.exists());
}

#[test]
fn raw_journal_cannot_read_artifacts_outside_its_directory() {
    let root = write_golden_directory();
    let path = root.path().join("capture-raw.json");
    let mut manifest: RawCaptureManifest =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest.takes[0].wav_file = "../other.wav".into();
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(
        io::process_capture_session(root.path(), &root.path().join("processed"))
            .unwrap_err()
            .contains("relative filenames")
    );
}

#[test]
fn selection_rejects_duplicate_unknown_and_incomplete_take_ids_before_writing() {
    let root = write_golden_directory();
    let raw: RawCaptureManifest =
        serde_json::from_slice(&std::fs::read(root.path().join("capture-raw.json")).unwrap())
            .unwrap();
    let first = raw.takes[0].take_id.clone();
    let second = raw.takes[1].take_id.clone();
    let cases = [
        (vec![first.clone(), first], "selected take ID is duplicated"),
        (
            vec![second.clone(), "unknown-take".into()],
            "selected take ID is unknown",
        ),
        (
            vec![second],
            "exactly one completed take per source/microphone pair",
        ),
    ];
    for (index, (selected, expected)) in cases.into_iter().enumerate() {
        let output = root.path().join(format!("invalid-selection-{index}"));
        let error = io::process_capture_session_with_selection(root.path(), &output, &selected)
            .unwrap_err();
        assert!(error.contains(expected), "{error}");
        assert!(!output.exists());
    }
}

#[test]
fn raw_take_repeat_index_must_belong_to_the_plan() {
    let root = write_golden_directory();
    let path = root.path().join("capture-raw.json");
    let mut manifest: RawCaptureManifest =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest.takes[0].repeat_index = manifest.plan.repeat_count;
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let output = root.path().join("out-of-domain-repeat");
    let error = io::process_capture_session(root.path(), &output).unwrap_err();
    assert!(
        error.contains("unknown, duplicate, or unidentified"),
        "{error}"
    );
    assert!(!output.exists());
}
