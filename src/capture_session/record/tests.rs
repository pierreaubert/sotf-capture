use super::*;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn setup() -> (tempfile::TempDir, ValidatedCaptureSession) {
    let root = tempfile::tempdir().unwrap();
    let mut plan: CaptureSessionPlan =
        serde_json::from_str(include_str!("../../../tests/fixtures/capture-session.json")).unwrap();
    plan.sweep.duration_secs = 0.05;
    for mic in &plan.microphones {
        std::fs::write(
            root.path().join(&mic.calibration_file),
            "20 0\n1000 1\n20000 0\n",
        )
        .unwrap();
    }
    (root, plan.validate().unwrap())
}

fn fake_capture(
    request: MultiCaptureRequest,
    _: &CancelFlag,
) -> Result<MultiCaptureResult, String> {
    assert_eq!(request.output_overrides.len(), 2);
    assert!(
        request
            .output_overrides
            .iter()
            .all(|region| region.channel == 0)
    );
    Ok(MultiCaptureResult {
        input_device_ids: request
            .inputs
            .iter()
            .map(|input| format!("id:{}", input.device))
            .collect(),
        output_device_id: "id:output".into(),
        sample_rate_hz: request.sample_rate_hz,
        recordings: request
            .inputs
            .iter()
            .map(|_| request.stimulus.clone())
            .collect(),
        input_sample_formats: vec!["F32".into(); request.inputs.len()],
        output_sample_format: "F32".into(),
    })
}

fn journal(directory: &Path) -> RawCaptureManifest {
    serde_json::from_slice(&std::fs::read(directory.join("capture-raw.json")).unwrap()).unwrap()
}

#[test]
fn all_mics_record_each_source_and_calibration_snapshots_are_frozen() {
    let (root, session) = setup();
    let output = root.path().join("recording");
    let cancel = Arc::new(AtomicBool::new(false));
    let mut sources = Vec::new();
    let manifest = record_with(
        &session,
        root.path(),
        &output,
        &cancel,
        |event| sources.push(event.source_id),
        |request, cancel| {
            // An external edit after acquisition starts must not change provenance.
            std::fs::write(
                root.path()
                    .join(&session.plan().microphones[0].calibration_file),
                "changed",
            )
            .unwrap();
            fake_capture(request, cancel)
        },
    )
    .unwrap();
    assert_eq!(sources, ["left", "right"]);
    assert_eq!(manifest.status, RawCaptureStatus::RawComplete);
    assert_eq!(manifest.takes.len(), 4);
    assert!(
        manifest
            .pending_processing
            .contains(&"clock_correction".to_owned())
    );
    for take in &manifest.takes {
        assert!(output.join(&take.wav_file).is_file());
        assert_eq!(take.clipped_samples, 0);
    }
    for calibration in &manifest.calibrations {
        let bytes = std::fs::read(output.join(&calibration.file)).unwrap();
        assert_eq!(calibration.sha256, super::super::calibration_sha256(&bytes));
        assert_ne!(bytes, b"changed");
    }
    assert_eq!(journal(&output).takes.len(), 4);
    assert!(
        !output.join("recordings.json").exists(),
        "raw journal must not masquerade as corrected import"
    );
}

#[test]
fn cancellation_retains_previous_source_and_explicit_status() {
    let (root, session) = setup();
    let output = root.path().join("recording");
    let cancel = Arc::new(AtomicBool::new(false));
    let error = record_with(
        &session,
        root.path(),
        &output,
        &cancel,
        |_| {},
        |request, flag| {
            flag.store(true, Ordering::Relaxed);
            fake_capture(request, flag)
        },
    )
    .unwrap_err();
    assert_eq!(error, "cancelled");
    let saved = journal(&output);
    assert_eq!(saved.status, RawCaptureStatus::Cancelled);
    assert_eq!(saved.takes.len(), 2);
}

#[test]
fn wrong_rate_fails_without_saving_a_successful_take() {
    let (root, session) = setup();
    let output = root.path().join("recording");
    let cancel = Arc::new(AtomicBool::new(false));
    assert!(
        record_with(
            &session,
            root.path(),
            &output,
            &cancel,
            |_| {},
            |request, flag| {
                let mut result = fake_capture(request, flag)?;
                result.sample_rate_hz = 44_100;
                Ok(result)
            }
        )
        .is_err()
    );
    let saved = journal(&output);
    assert_eq!(saved.status, RawCaptureStatus::Failed);
    assert!(saved.takes.is_empty());
}

#[test]
fn calibration_failure_precedes_audio_and_directory_creation() {
    let (root, session) = setup();
    let output = root.path().join("recording");
    std::fs::remove_file(
        root.path()
            .join(&session.plan().microphones[1].calibration_file),
    )
    .unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    assert!(
        record_with(
            &session,
            root.path(),
            &output,
            &cancel,
            |_| {},
            |_, _| panic!("audio must not start")
        )
        .is_err()
    );
    assert!(!output.exists());
}

#[test]
fn existing_directory_is_never_overwritten() {
    let (root, session) = setup();
    let output = root.path().join("recording");
    std::fs::create_dir(&output).unwrap();
    std::fs::write(output.join("keep"), "existing recording").unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    assert!(
        record_with(
            &session,
            root.path(),
            &output,
            &cancel,
            |_| {},
            |_, _| panic!("audio must not start")
        )
        .is_err()
    );
    assert_eq!(
        std::fs::read_to_string(output.join("keep")).unwrap(),
        "existing recording"
    );
}

#[test]
fn cancelled_repeated_parent_processes_an_explicit_full_matrix_and_keeps_lineage() {
    use crate::capture_session::clock::io::process_capture_session_with_selection;

    let root = tempfile::tempdir().unwrap();
    let mut plan: CaptureSessionPlan =
        serde_json::from_str(include_str!("../../../tests/fixtures/capture-session.json")).unwrap();
    plan.sweep.duration_secs = 0.05;
    plan.sources.truncate(1);
    plan.repeat_count = 2;
    for mic in &plan.microphones {
        std::fs::write(
            root.path().join(&mic.calibration_file),
            "20 0\n1000 1\n20000 0\n",
        )
        .unwrap();
    }
    let session = plan.validate().unwrap();
    let raw = root.path().join("raw-session");
    let cancel = Arc::new(AtomicBool::new(false));
    let error = record_with(
        &session,
        root.path(),
        &raw,
        &cancel,
        |_| {},
        |request, flag| {
            let captured = fake_capture(request, flag)?;
            flag.store(true, Ordering::Relaxed);
            Ok(captured)
        },
    )
    .unwrap_err();
    assert_eq!(error, "cancelled");
    let parent_bytes = std::fs::read(raw.join("capture-raw.json")).unwrap();
    let parent: RawCaptureManifest = serde_json::from_slice(&parent_bytes).unwrap();
    assert_eq!(parent.status, RawCaptureStatus::Cancelled);
    assert_eq!(parent.plan.repeat_count, 2);
    assert_eq!(parent.takes.len(), parent.plan.microphones.len());
    assert!(parent.takes.iter().all(|take| take.repeat_index == 0));

    let selected = parent
        .takes
        .iter()
        .map(|take| take.take_id.clone())
        .collect::<Vec<_>>();
    let output = root.path().join("processed-session");
    let report = process_capture_session_with_selection(&raw, &output, &selected).unwrap();
    assert_eq!(report.raw_status, Some(RawCaptureStatus::Cancelled));
    assert_eq!(
        report.selected_take_ids.as_deref(),
        Some(selected.as_slice())
    );
    assert_eq!(
        report.recording_manifest.as_deref(),
        Some("recordings.json")
    );
    assert_eq!(
        std::fs::read(output.join("capture-raw.json")).unwrap(),
        parent_bytes
    );

    let handoff: autoeq::capture_handoff::CaptureHandoff = serde_json::from_slice(
        &std::fs::read(output.join(autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME)).unwrap(),
    )
    .unwrap();
    handoff.validate().unwrap();
    assert_eq!(
        handoff.completion,
        autoeq::capture_handoff::CaptureCompletion::Cancelled
    );
    assert_eq!(handoff.repeat_count, 2);
    assert_eq!(
        handoff.selected_take_ids.as_deref(),
        Some(selected.as_slice())
    );
    assert_eq!(handoff.takes.len(), parent.takes.len());
    let parent_identity = handoff
        .artifacts
        .iter()
        .find(|artifact| artifact.file == "capture-raw.json")
        .unwrap();
    assert_eq!(parent_identity.sha256, handoff.session_id);
    for (processed, raw_take) in handoff.takes.iter().zip(&parent.takes) {
        assert_eq!(processed.take_id, raw_take.take_id);
        assert_eq!(processed.repeat_index, raw_take.repeat_index);
        assert_eq!(
            std::fs::read(output.join(&processed.raw_audio_file)).unwrap(),
            std::fs::read(raw.join(&raw_take.wav_file)).unwrap()
        );
    }

    // Exercise the strict public consumer with a cancelled but explicitly
    // selected full source/microphone matrix.
    autoeq::roomeq::load_config(&output.join("recordings.json"), None).unwrap();
    let mut changed_parent = parent_bytes;
    changed_parent[0] ^= 1;
    std::fs::write(output.join("capture-raw.json"), changed_parent).unwrap();
    assert!(autoeq::roomeq::load_config(&output.join("recordings.json"), None).is_err());
}

fn aux_session() -> (
    tempfile::TempDir,
    crate::capture_session::ValidatedCaptureSession,
) {
    let root = tempfile::tempdir().unwrap();
    let mut plan: crate::capture_session::CaptureSessionPlan =
        serde_json::from_str(include_str!("../../../tests/fixtures/capture-session.json")).unwrap();
    plan.probe = Some(crate::capture_session::CaptureProbe {
        probe_duration_ms: 1000.0,
        silence_duration_ms: 500.0,
        signal_level_db: -6.0,
    });
    plan.bass_anchor = Some(crate::capture_session::CaptureBassAnchor {
        bass_freq_hz: 30.0,
        bass_duration_s: 2.0,
        fade_ms: 50.0,
        num_windows: 8,
        silence_duration_ms: 500.0,
        signal_level_db: -6.0,
    });
    plan.loopback = Some(crate::capture_session::CaptureLoopback {
        device: plan.microphones[0].device.clone(),
        input_channel: 7,
    });
    (root, plan.validate().unwrap())
}

fn aux_manifest(session: &crate::capture_session::ValidatedCaptureSession) -> RawCaptureManifest {
    let stimulus = crate::capture_session::protocol::prepare_capture_stimulus(session).unwrap();
    RawCaptureManifest {
        version: 1,
        plan: session.plan().clone(),
        probe_pass: None,
        bass_anchor_pass: None,
        stimulus: stimulus.layout,
        timing_reference_output_channel: Some(0),
        status: RawCaptureStatus::RawComplete,
        pending_processing: Vec::new(),
        calibrations: Vec::new(),
        takes: Vec::new(),
        loopback_files: Vec::new(),
        error: None,
    }
}

fn canned_probe() -> crate::signal_recorder::ProbeDelayResults {
    crate::signal_recorder::ProbeDelayResults {
        channels: vec![crate::signal_recorder::ProbeDelayChannelResult {
            channel_name: "left".into(),
            channel_index: 0,
            arrival_ms: 1.5,
            gain_db: -3.0,
            snr_db: 40.0,
        }],
        sample_rate: 48000,
        alignment_delays_ms: vec![0.5],
    }
}

fn canned_anchor() -> crate::signal_recorder::BassAnchorResults {
    crate::signal_recorder::BassAnchorResults {
        channels: vec![crate::signal_recorder::BassAnchorChannelResult {
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
    }
}

#[test]
fn auxiliary_passes_journal_results_with_plan_routing() {
    let (root, session) = aux_session();
    let output = root.path().join("aux-session");
    std::fs::create_dir(&output).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let manifest = run_auxiliary_passes(
        &session,
        &output,
        &cancel,
        aux_manifest(&session),
        |channels, names, rate, duration, silence, _out, _input, _ch, wav, _level, _cancel| {
            assert_eq!(channels, &[0, 1]);
            assert_eq!(names, &["left".to_string(), "right".to_string()]);
            assert_eq!(rate, 48000);
            assert_eq!(duration, 1000.0);
            assert_eq!(silence, 500.0);
            assert_eq!(
                wav.file_name().and_then(|name| name.to_str()),
                Some("probe.wav")
            );
            Ok(canned_probe())
        },
        |channels,
         names,
         rate,
         freq,
         duration,
         fade,
         windows,
         _silence,
         _out,
         _input,
         _ch,
         loopback,
         wav,
         _level,
         _cancel| {
            assert_eq!(channels, &[0, 1]);
            assert_eq!(names, &["left".to_string(), "right".to_string()]);
            assert_eq!(rate, 48000);
            assert_eq!(freq, 30.0);
            assert_eq!(duration, 2.0);
            assert_eq!(fade, 50.0);
            assert_eq!(windows, 8);
            assert_eq!(loopback, Some(7));
            assert_eq!(
                wav.file_name().and_then(|name| name.to_str()),
                Some("bass-anchor.wav")
            );
            Ok(canned_anchor())
        },
    )
    .unwrap();
    assert_eq!(manifest.status, RawCaptureStatus::RawComplete);
    let probe = manifest.probe_pass.as_ref().unwrap();
    assert_eq!(probe.wav_file, "probe.wav");
    assert_eq!(probe.microphone_id, "seat-left");
    assert_eq!(probe.results.alignment_delays_ms, vec![0.5]);
    let anchor = manifest.bass_anchor_pass.as_ref().unwrap();
    assert_eq!(anchor.wav_file, "bass-anchor.wav");
    assert_eq!(anchor.loopback_channel, Some(7));
    assert_eq!(anchor.results.bass_freq_hz, 30.0);
    let saved = journal(&output);
    assert!(saved.probe_pass.is_some());
    assert!(saved.bass_anchor_pass.is_some());
}

#[test]
fn auxiliary_failure_marks_the_session_failed() {
    let (root, session) = aux_session();
    let output = root.path().join("aux-session");
    std::fs::create_dir(&output).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let error = run_auxiliary_passes(
        &session,
        &output,
        &cancel,
        aux_manifest(&session),
        |_, _, _, _, _, _, _, _, _, _, _| Err("no probe tone".to_string()),
        |_, _, _, _, _, _, _, _, _, _, _, _, _, _, _| Ok(canned_anchor()),
    )
    .unwrap_err();
    assert!(error.contains("no probe tone"), "{error}");
    let saved = journal(&output);
    assert_eq!(saved.status, RawCaptureStatus::Failed);
    assert!(saved.probe_pass.is_none());
}

#[test]
fn auxiliary_passes_honor_cancellation() {
    let (root, session) = aux_session();
    let output = root.path().join("aux-session");
    std::fs::create_dir(&output).unwrap();
    let cancel = Arc::new(AtomicBool::new(true));
    let error = run_auxiliary_passes(
        &session,
        &output,
        &cancel,
        aux_manifest(&session),
        |_, _, _, _, _, _, _, _, _, _, _| Ok(canned_probe()),
        |_, _, _, _, _, _, _, _, _, _, _, _, _, _, _| Ok(canned_anchor()),
    )
    .unwrap_err();
    assert_eq!(error, "cancelled");
    assert_eq!(journal(&output).status, RawCaptureStatus::Cancelled);
}

#[test]
fn ctc_sessions_record_one_loopback_take_per_source() {
    let (root, plain) = setup();
    let mut plan = plain.plan().clone();
    plan.loopback = Some(crate::capture_session::CaptureLoopback {
        device: plan.microphones[0].device.clone(),
        input_channel: 7,
    });
    plan.ctc = Some(crate::capture_session::CaptureCtc {
        left_ear: plan.microphones[0].id.clone(),
        right_ear: plan.microphones[1].id.clone(),
        head_position: crate::capture_session::CaptureHeadPosition {
            id: "head-0".into(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            yaw_deg: 0.0,
        },
    });
    let session = plan.validate().unwrap();
    let output = root.path().join("session");
    let cancel = Arc::new(AtomicBool::new(false));
    let manifest = record_with(
        &session,
        root.path(),
        &output,
        &cancel,
        |_| {},
        fake_capture,
    )
    .unwrap();
    assert_eq!(manifest.status, RawCaptureStatus::RawComplete);
    assert_eq!(manifest.loopback_files.len(), session.plan().sources.len());
    for loopback in &manifest.loopback_files {
        assert!(output.join(&loopback.wav_file).is_file());
        assert!(loopback.samples > 0);
        assert_eq!(
            loopback.device_id,
            format!("id:{}", session.plan().loopback.as_ref().unwrap().device)
        );
    }
    let saved = journal(&output);
    assert_eq!(saved.loopback_files.len(), session.plan().sources.len());
}

#[test]
fn sessions_without_auxiliary_sections_skip_the_runners() {
    let (_root, session) = setup();
    let output = tempfile::tempdir().unwrap();
    let out = output.path().join("aux-session");
    std::fs::create_dir(&out).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let manifest = run_auxiliary_passes(
        &session,
        &out,
        &cancel,
        aux_manifest(&session),
        |_, _, _, _, _, _, _, _, _, _, _| -> Result<_, String> {
            unreachable!("probe must not run without a plan section")
        },
        |_, _, _, _, _, _, _, _, _, _, _, _, _, _, _| -> Result<_, String> {
            unreachable!("anchor must not run without a plan section")
        },
    )
    .unwrap();
    assert!(manifest.probe_pass.is_none());
    assert!(manifest.bass_anchor_pass.is_none());
}
