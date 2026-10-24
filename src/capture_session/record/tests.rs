use super::*;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

#[path = "checkpoint_tests.rs"]
mod checkpoint_tests;

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

#[test]
fn session_plan_accepts_large_file_backed_takes_within_the_frame_limit() {
    let (_, session) = setup();
    let mut plan = session.plan().clone();
    let microphone = plan.microphones[0].clone();
    plan.microphones = (0_u16..32)
        .map(|channel| {
            let mut mic = microphone.clone();
            mic.id = format!("mic-{channel}");
            mic.input_channel = channel;
            mic
        })
        .collect();
    plan.loopback = Some(crate::capture_session::CaptureLoopback {
        device: microphone.device,
        input_channel: 63,
    });
    plan.sweep.duration_secs = 200.0;
    assert!(plan.validate().is_ok());
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "The fixture implements the capture callback contract, which transfers its request by value."
)]
#[expect(
    clippy::unnecessary_wraps,
    reason = "The fixture implements the fallible capture callback contract."
)]
fn fake_capture(
    request: MultiCaptureRequest,
    _: &CancelFlag,
) -> Result<MultiCaptureResult, String> {
    // Wake-up probes carry no chirp overrides and echo back for lag lock;
    // real takes always bracket with start+end chirps on channel 0.
    if !request.output_overrides.is_empty() {
        assert_eq!(request.output_overrides.len(), 2);
        assert!(
            request
                .output_overrides
                .iter()
                .all(|region| region.channel == 0)
        );
    }
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
fn checkpoints_are_saved_before_wake_and_every_repeated_source() {
    let (root, session) = setup();
    let mut plan = session.plan().clone();
    plan.repeat_count = 2;
    let session = plan.validate().unwrap();
    let output = root.path().join("recording");
    let cancel = Arc::new(AtomicBool::new(false));
    let mut phases = Vec::new();
    let result = record_with(
        &session,
        root.path(),
        &output,
        &cancel,
        |_| {},
        |request, flag| {
            let current = journal(&output);
            let checkpoint = current.checkpoint.as_ref().unwrap();
            assert_eq!(current.status, RawCaptureStatus::Capturing);
            assert_eq!(
                checkpoint.completed_take_ids,
                current
                    .takes
                    .iter()
                    .map(|take| take.take_id.clone())
                    .collect::<Vec<_>>()
            );
            if request.output_overrides.is_empty() {
                assert_eq!(checkpoint.phase, CapturePhase::Waking);
            } else {
                let CapturePhase::Sweep {
                    source_id,
                    repeat_index,
                } = &checkpoint.phase
                else {
                    panic!("expected sweep checkpoint");
                };
                let source = session
                    .plan()
                    .sources
                    .iter()
                    .find(|source| &source.id == source_id)
                    .unwrap();
                assert_eq!(source.output_channel, request.output_channel);
                phases.push((source_id.clone(), *repeat_index));
            }
            fake_capture(request, flag)
        },
    )
    .unwrap();
    assert_eq!(
        phases,
        [
            ("left".into(), 0),
            ("right".into(), 0),
            ("left".into(), 1),
            ("right".into(), 1)
        ]
    );
    let checkpoint = result.checkpoint.as_ref().unwrap();
    assert_eq!(checkpoint.phase, CapturePhase::Complete);
    assert_eq!(checkpoint.completed_take_ids.len(), 8);
    assert_eq!(journal(&output).checkpoint, result.checkpoint);
}

#[test]
fn completed_recording_returns_the_same_frozen_reports_as_its_journal() {
    let (root, session) = setup();
    let output = root.path().join("recording");
    let flag = Arc::new(AtomicBool::new(false));
    let result = record_with(
        &session,
        root.path(),
        &output,
        &flag,
        |_| {},
        |request, cancel| {
            if request.output_overrides.is_empty() {
                crate::capture_session::loss_evidence::fixture(
                    &output,
                    "capture-loss-prior-wake.json",
                );
            }
            fake_capture(request, cancel)
        },
    )
    .unwrap();
    let saved = journal(&output);
    assert_eq!(result.status, RawCaptureStatus::RawComplete);
    assert_eq!(result.transport_evidence.len(), 1);
    assert_eq!(
        serde_json::to_value(result.transport_evidence).unwrap(),
        serde_json::to_value(saved.transport_evidence).unwrap()
    );
}

#[test]
fn recording_journals_bind_reports_even_when_cancellation_has_no_report_in_its_error() {
    for reason in ["capture lost input frames", "cancelled"] {
        let (root, session) = setup();
        let output = root.path().join("recording");
        let flag = Arc::new(AtomicBool::new(false));
        let error = record_with(
            &session,
            root.path(),
            &output,
            &flag,
            |_| {},
            |request, cancel| {
                if request.output_overrides.is_empty() {
                    fake_capture(request, cancel)
                } else {
                    crate::capture_session::loss_evidence::fixture(
                        &output,
                        "capture-loss-acquisition.json",
                    );
                    Err(reason.into())
                }
            },
        )
        .unwrap_err();
        assert_eq!(error, reason);
        let saved = journal(&output);
        assert_eq!(saved.transport_evidence.len(), 1);
        let snapshot = &saved.transport_evidence[0];
        assert_eq!(snapshot.file, "capture-loss-acquisition.json");
        assert_eq!(
            snapshot.sha256,
            crate::capture_session::calibration_sha256(
                &std::fs::read(output.join(&snapshot.file)).unwrap()
            )
        );
        assert_eq!(
            saved.status,
            if reason == "cancelled" {
                RawCaptureStatus::Cancelled
            } else {
                RawCaptureStatus::Failed
            }
        );
        assert_eq!(saved.takes.len(), 0);
    }
}

#[cfg(feature = "capture-io")]
fn fake_file_capture(
    root: &Path,
    request: MultiCaptureRequest,
    cancel: &CancelFlag,
) -> Result<crate::signal_recorder::multi_capture::FileCaptureResult, String> {
    let captured = fake_capture(request, cancel)?;
    Ok(crate::signal_recorder::multi_capture::FileCaptureResult {
        recordings: captured
            .recordings
            .into_iter()
            .map(|samples| {
                crate::signal_recorder::multi_capture::CapturedFile::fixture(
                    root,
                    captured.sample_rate_hz,
                    &samples,
                )
            })
            .collect(),
        input_device_ids: captured.input_device_ids,
        output_device_id: captured.output_device_id,
        sample_rate_hz: captured.sample_rate_hz,
        input_sample_formats: captured.input_sample_formats,
        output_sample_format: captured.output_sample_format,
    })
}

#[cfg(feature = "capture-io")]
#[test]
fn file_backed_session_publishes_exact_wave_metadata_without_private_temporaries() {
    let (root, session) = setup();
    let output = root.path().join("streamed");
    let cancel = Arc::new(AtomicBool::new(false));
    let manifest = record_with(
        &session,
        root.path(),
        &output,
        &cancel,
        |_| {},
        |request, cancel| fake_file_capture(&output, request, cancel),
    )
    .unwrap();
    assert_eq!(manifest.status, RawCaptureStatus::RawComplete);
    assert_eq!(manifest.takes.len(), 4);
    for take in &manifest.takes {
        let mut reader = hound::WavReader::open(output.join(&take.wav_file)).unwrap();
        assert_eq!(reader.spec().sample_rate, session.plan().sample_rate_hz);
        assert_eq!(usize::try_from(reader.duration()).unwrap(), take.samples);
        let mut peak = 0.0_f32;
        let mut clipped = 0;
        for sample in reader.samples::<f32>() {
            let sample = sample.unwrap();
            peak = peak.max(sample.abs());
            clipped += usize::from(sample.abs() >= 1.0);
        }
        assert_eq!(peak, take.peak_amplitude);
        assert_eq!(clipped, take.clipped_samples);
    }
    assert!(std::fs::read_dir(&output).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with('.')
    }));
}

#[cfg(feature = "capture-io")]
#[test]
fn file_backed_session_keeps_completed_sources_on_later_failure_or_cancellation() {
    for cancelled in [false, true] {
        let (root, session) = setup();
        let output = root.path().join("streamed");
        let cancel = Arc::new(AtomicBool::new(false));
        let mut takes = 0;
        let error = record_with(
            &session,
            root.path(),
            &output,
            &cancel,
            |_| {},
            |request, flag| {
                let is_take = !request.output_overrides.is_empty();
                if is_take {
                    takes += 1;
                }
                if takes == 2 {
                    return Err("injected disk write failure".into());
                }
                let result = fake_file_capture(&output, request, flag)?;
                if is_take && cancelled {
                    flag.store(true, Ordering::Relaxed);
                }
                Ok(result)
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            if cancelled {
                "cancelled"
            } else {
                "injected disk write failure"
            }
        );
        let saved = journal(&output);
        assert_eq!(
            saved.status,
            if cancelled {
                RawCaptureStatus::Cancelled
            } else {
                RawCaptureStatus::Failed
            }
        );
        assert_eq!(saved.takes.len(), session.plan().microphones.len());
        for take in saved.takes {
            assert!(hound::WavReader::open(output.join(take.wav_file)).is_ok());
        }
        assert!(!output.join("recordings.json").exists());
        assert!(std::fs::read_dir(&output).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with('.')
        }));
    }
}

#[cfg(feature = "capture-io")]
#[test]
fn file_backed_publication_collision_preserves_destination_and_marks_journal_failed() {
    let (root, session) = setup();
    let output = root.path().join("streamed");
    let cancel = Arc::new(AtomicBool::new(false));
    let collision = output.join("repeat-00-source-00-mic-00-raw.wav");
    let error = record_with(
        &session,
        root.path(),
        &output,
        &cancel,
        |_| {},
        |request, flag| {
            if !request.output_overrides.is_empty() {
                std::fs::write(&collision, b"existing sentinel").unwrap();
            }
            fake_file_capture(&output, request, flag)
        },
    )
    .unwrap_err();
    assert!(error.contains("cannot publish captured WAV"), "{error}");
    assert_eq!(std::fs::read(&collision).unwrap(), b"existing sentinel");
    let saved = journal(&output);
    assert_eq!(saved.status, RawCaptureStatus::Failed);
    assert!(saved.takes.is_empty());
    assert!(std::fs::read_dir(&output).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with('.')
    }));
}

#[cfg(feature = "capture-io")]
#[test]
fn file_backed_crash_before_publication_leaves_a_nonterminal_journal() {
    const CHILD_ROOT: &str = "SOTF_STREAM_CRASH_TEST_ROOT";
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        let root = std::path::PathBuf::from(root);
        let plan: CaptureSessionPlan =
            serde_json::from_slice(&std::fs::read(root.join("replay-plan.json")).unwrap()).unwrap();
        let session = plan.validate().unwrap();
        let output = root.join("crashed");
        let cancel = Arc::new(AtomicBool::new(false));
        record_with(
            &session,
            &root,
            &output,
            &cancel,
            |_| {},
            |request, flag| {
                let is_take = !request.output_overrides.is_empty();
                let captured = fake_file_capture(&output, request, flag)?;
                if is_take {
                    // Finalized WAVs exist privately, but their paths have not been
                    // published or added to the session journal. Skip destructors.
                    std::process::exit(73);
                }
                Ok(captured)
            },
        )
        .unwrap();
        panic!("crash point was not reached");
    }
    let (root, session) = setup();
    std::fs::write(
        root.path().join("replay-plan.json"),
        serde_json::to_vec(session.plan()).unwrap(),
    )
    .unwrap();
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("capture_session::record::tests::file_backed_crash_before_publication_leaves_a_nonterminal_journal")
        .env(CHILD_ROOT, root.path()).output().unwrap();
    assert_eq!(child.status.code(), Some(73), "{child:?}");
    let crashed = root.path().join("crashed");
    let saved = journal(&crashed);
    assert_eq!(saved.status, RawCaptureStatus::Capturing);
    assert!(matches!(
        saved.checkpoint.as_ref().unwrap().phase,
        CapturePhase::Sweep { .. }
    ));
    assert!(saved.takes.is_empty());
    assert!(std::fs::read_dir(&crashed).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("repeat-")
    }));
    let refused = root.path().join("refused");
    assert!(super::super::clock::io::process_capture_session(&crashed, &refused).is_err());
    assert!(
        !refused
            .join(autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME)
            .exists()
    );
    let retry = root.path().join("retry");
    let cancel = Arc::new(AtomicBool::new(false));
    let completed = record_with(
        &session,
        root.path(),
        &retry,
        &cancel,
        |_| {},
        |request, flag| fake_file_capture(&retry, request, flag),
    )
    .unwrap();
    assert_eq!(completed.status, RawCaptureStatus::RawComplete);
    assert_eq!(completed.takes.len(), 4);
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
fn bench_and_loopback_evidence_survive_source_mutation_and_bundle_moves() {
    let (root, original) = setup();
    let mut plan = original.plan().clone();
    plan.sources.truncate(1);
    plan.localization_calibration_file = Some("bench.json".into());
    plan.loopback = Some(crate::capture_session::CaptureLoopback {
        device: plan.microphones[0].device.clone(),
        input_channel: 7,
    });
    let bytes = br#"{"schema":"sotf-capture/localization-calibration","schema_version":1,"latency_samples_by_rate_hz":{"48000":0},"mic_bias_measured":true,"mic_bias_correction_m":0.004}"#;
    std::fs::write(root.path().join("bench.json"), bytes).unwrap();
    let session = plan.validate().unwrap();
    let raw = root.path().join("raw");
    let cancel = Arc::new(AtomicBool::new(false));
    let manifest = record_with(
        &session,
        root.path(),
        &raw,
        &cancel,
        |_| {},
        |request, flag| {
            std::fs::write(root.path().join("bench.json"), b"changed after preflight").unwrap();
            fake_capture(request, flag)
        },
    )
    .unwrap();
    let snapshot = manifest.localization_calibration.as_ref().unwrap();
    assert_eq!(std::fs::read(raw.join(&snapshot.file)).unwrap(), bytes);
    assert_eq!(snapshot.sha256, super::super::calibration_sha256(bytes));
    assert_eq!(
        manifest.plan.localization_calibration_file.as_deref(),
        Some(Path::new(&snapshot.file))
    );
    std::fs::remove_file(root.path().join("bench.json")).unwrap();
    let moved = root.path().join("moved-raw");
    std::fs::rename(&raw, &moved).unwrap();
    let output = root.path().join("processed");
    let report = super::super::clock::io::process_capture_session(&moved, &output).unwrap();
    let handoff: autoeq::capture_handoff::CaptureHandoff = serde_json::from_slice(
        &std::fs::read(output.join(autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME)).unwrap(),
    )
    .unwrap();
    for file in std::iter::once(snapshot.file.as_str()).chain(
        manifest
            .loopback_files
            .iter()
            .map(|take| take.wav_file.as_str()),
    ) {
        assert_eq!(
            std::fs::read(output.join(file)).unwrap(),
            std::fs::read(moved.join(file)).unwrap()
        );
        assert!(
            handoff
                .artifacts
                .iter()
                .any(|artifact| artifact.file == file),
            "{file}"
        );
    }
    let again = super::super::clock::io::process_capture_session(
        &moved,
        &root.path().join("processed-again"),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(report.localization_report).unwrap(),
        serde_json::to_value(again.localization_report).unwrap()
    );
    std::fs::write(moved.join(&snapshot.file), b"tampered").unwrap();
    let rejected = root.path().join("rejected");
    assert!(super::super::clock::io::process_capture_session(&moved, &rejected).is_err());
    assert!(!rejected.exists());
}

#[test]
fn loss_evidence_reference_survives_in_failed_and_cancelled_journals() {
    for (reason, status) in [
        (
            "capture lost input frames; acquisition evidence: capture-loss-test.json",
            RawCaptureStatus::Failed,
        ),
        (
            "cancelled; acquisition evidence: capture-loss-test.json",
            RawCaptureStatus::Cancelled,
        ),
    ] {
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
                if request.output_overrides.is_empty() {
                    fake_capture(request, flag)
                } else {
                    Err(reason.into())
                }
            },
        )
        .unwrap_err();
        assert_eq!(error, reason);
        let saved = journal(&output);
        assert_eq!(saved.status, status);
        assert_eq!(saved.error.as_deref(), Some(reason));
        assert_eq!(saved.takes.len(), 0);
        assert!(!output.join("handoff.json").exists());
    }
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
            // Cancel on take calls only: the wake-up probe must run first.
            if !request.output_overrides.is_empty() {
                flag.store(true, Ordering::Relaxed);
            }
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
            |_, _| -> Result<MultiCaptureResult, String> { panic!("audio must not start") }
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
            |_, _| -> Result<MultiCaptureResult, String> { panic!("audio must not start") }
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
            // Cancel after the first take completes, not during wake-up.
            let is_take = !request.output_overrides.is_empty();
            let captured = fake_capture(request, flag)?;
            if is_take {
                flag.store(true, Ordering::Relaxed);
            }
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
        checkpoint: None,
        transport_evidence: Vec::new(),
        localization_calibration: None,
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
fn auxiliary_cancellation_registers_its_report_before_saving_terminal_status() {
    let (root, session) = aux_session();
    let output = root.path().join("aux-session");
    std::fs::create_dir(&output).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let error = run_auxiliary_passes(
        &session,
        &output,
        &cancel,
        aux_manifest(&session),
        |_, _, _, _, _, _, _, _, _, _, _| {
            crate::capture_session::loss_evidence::fixture(&output, "capture-loss-probe.json");
            Err("cancelled".into())
        },
        |_, _, _, _, _, _, _, _, _, _, _, _, _, _, _| panic!("anchor must not run"),
    )
    .unwrap_err();
    assert_eq!(error, "cancelled");
    let saved = journal(&output);
    assert_eq!(saved.status, RawCaptureStatus::Cancelled);
    assert_eq!(saved.transport_evidence.len(), 1);
    assert_eq!(saved.transport_evidence[0].file, "capture-loss-probe.json");
    assert!(saved.probe_pass.is_none());
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
            let current = journal(&output);
            assert_eq!(current.status, RawCaptureStatus::Capturing);
            let checkpoint = current.checkpoint.as_ref().unwrap();
            assert_eq!(checkpoint.phase, CapturePhase::Probe);
            assert!(!checkpoint.probe_completed && !checkpoint.bass_anchor_completed);
            assert!(current.probe_pass.is_none() && current.bass_anchor_pass.is_none());
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
            let current = journal(&output);
            assert_eq!(current.status, RawCaptureStatus::Capturing);
            let checkpoint = current.checkpoint.as_ref().unwrap();
            assert_eq!(checkpoint.phase, CapturePhase::BassAnchor);
            assert!(checkpoint.probe_completed && !checkpoint.bass_anchor_completed);
            assert!(current.probe_pass.is_some() && current.bass_anchor_pass.is_none());
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
    let checkpoint = manifest.checkpoint.as_ref().unwrap();
    assert_eq!(checkpoint.phase, CapturePhase::Complete);
    assert!(checkpoint.probe_completed && checkpoint.bass_anchor_completed);
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
    let (root, original) = setup();
    let mut plan = original.plan().clone();
    plan.microphones[1].device = plan.microphones[0].device.clone();
    plan.microphones[1].input_channel = 1;
    for input_channel in [2, 3] {
        let mut mic = plan.microphones[0].clone();
        mic.id = format!("extra-{input_channel}");
        mic.input_channel = input_channel;
        plan.microphones.push(mic);
    }
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
    let mut sweep_requests = 0;
    let manifest = record_with(
        &session,
        root.path(),
        &output,
        &cancel,
        |_| {},
        |request, cancel| {
            if !request.output_overrides.is_empty() {
                assert_eq!(request.inputs.len(), 5);
                assert_eq!(request.inputs[4].channel, 7);
                sweep_requests += 1;
            }
            fake_capture(request, cancel)
        },
    )
    .unwrap();
    assert_eq!(sweep_requests, session.plan().sources.len());
    assert_eq!(manifest.status, RawCaptureStatus::RawComplete);
    assert_eq!(manifest.takes.len(), session.plan().sources.len() * 4);
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
fn declared_loopback_records_takes_without_ctc() {
    let (root, original) = setup();
    let mut plan = original.plan().clone();
    plan.loopback = Some(crate::capture_session::CaptureLoopback {
        device: plan.microphones[0].device.clone(),
        input_channel: 7,
    });
    assert!(plan.ctc.is_none());
    let session = plan.validate().unwrap();
    let output = root.path().join("session");
    let cancel = Arc::new(AtomicBool::new(false));
    let manifest = record_with(
        &session,
        root.path(),
        &output,
        &cancel,
        |_| {},
        |request, cancel| {
            if !request.output_overrides.is_empty() {
                assert_eq!(request.inputs.len(), 3);
                assert_eq!(request.inputs[2].channel, 7);
            }
            fake_capture(request, cancel)
        },
    )
    .unwrap();
    assert_eq!(manifest.status, RawCaptureStatus::RawComplete);
    assert_eq!(manifest.loopback_files.len(), session.plan().sources.len());
    for loopback in &manifest.loopback_files {
        assert!(output.join(&loopback.wav_file).is_file());
        assert!(loopback.samples > 0);
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

fn silent_result(request: &MultiCaptureRequest) -> MultiCaptureResult {
    MultiCaptureResult {
        input_device_ids: request
            .inputs
            .iter()
            .map(|input| format!("id:{}", input.device))
            .collect(),
        output_device_id: "id:output".into(),
        sample_rate_hz: request.sample_rate_hz,
        recordings: vec![vec![0.0; request.stimulus.len()]; request.inputs.len()],
        input_sample_formats: vec!["F32".into(); request.inputs.len()],
        output_sample_format: "F32".into(),
    }
}

#[test]
fn wake_up_rides_through_sleeping_first_captures() {
    use std::cell::Cell;

    let (root, session) = setup();
    let output = root.path().join("recording");
    let cancel = Arc::new(AtomicBool::new(false));
    let calls = Cell::new(0_u32);
    // The first two backend calls render silence (USB asleep); the wake-up
    // burns them and the real takes record clean.
    let manifest = record_with(
        &session,
        root.path(),
        &output,
        &cancel,
        |_| {},
        |request, flag| {
            calls.set(calls.get() + 1);
            if calls.get() <= 2 {
                Ok(silent_result(&request))
            } else {
                fake_capture(request, flag)
            }
        },
    )
    .expect("wake-up must absorb the sleeping captures");
    assert_eq!(manifest.status, RawCaptureStatus::RawComplete);
    assert_eq!(manifest.takes.len(), 4);
    assert!(calls.get() >= 3);
}

#[test]
fn wake_up_budget_exhaustion_proceeds_without_error() {
    use std::time::{Duration, Instant};

    let (_root, session) = setup();
    let cancel = Arc::new(AtomicBool::new(false));
    // A chain that never answers: the budget bounds the wait and the call
    // still returns Ok — the first real take owns the refusal.
    let started = Instant::now();
    wake_session_chain(
        &mut |request, _| Ok(silent_result(&request)),
        &session,
        &cancel,
        Duration::from_millis(300),
    )
    .expect("exhaustion must proceed, not fail");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "budget must bound the wait"
    );
}

#[test]
fn wake_up_honors_cancel_immediately() {
    use std::time::Duration;

    let (_root, session) = setup();
    let cancel = Arc::new(AtomicBool::new(true));
    let error = wake_session_chain(
        &mut |request, flag| fake_capture(request, flag),
        &session,
        &cancel,
        Duration::from_secs(5),
    )
    .unwrap_err();
    assert_eq!(error, "cancelled");
}
