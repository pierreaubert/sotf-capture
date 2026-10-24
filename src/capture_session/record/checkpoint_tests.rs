//! Crash and publication-failure boundaries after committed sweeps.

// Rust guideline compliant 2026-02-21

use super::*;

fn prepare() -> (tempfile::TempDir, ValidatedCaptureSession) {
    let (root, session) = aux_session();
    let mut plan = session.plan().clone();
    plan.sweep.duration_secs = 0.05;
    for mic in &plan.microphones {
        std::fs::write(
            root.path().join(&mic.calibration_file),
            b"20 0\n1000 0\n20000 0\n",
        )
        .unwrap();
    }
    (root, plan.validate().unwrap())
}

fn sweeps(root: &Path, session: &ValidatedCaptureSession) -> RawCaptureManifest {
    let flag = Arc::new(AtomicBool::new(false));
    let manifest = record_with(
        session,
        root,
        &root.join("capture"),
        &flag,
        |_| {},
        fake_capture,
    )
    .unwrap();
    assert_eq!(manifest.status, RawCaptureStatus::Capturing);
    assert_eq!(
        manifest.checkpoint.as_ref().unwrap().phase,
        CapturePhase::AwaitingAuxiliary
    );
    manifest
}

fn assert_interrupted(root: &Path, session: &ValidatedCaptureSession) {
    let capture = root.join("capture");
    let saved = journal(&capture);
    assert_eq!(saved.status, RawCaptureStatus::Capturing);
    assert_eq!(
        saved.takes.len(),
        session.plan().sources.len() * session.plan().microphones.len()
    );
    let checkpoint = saved.checkpoint.as_ref().unwrap();
    assert_eq!(checkpoint.phase, CapturePhase::Probe);
    assert_eq!(
        checkpoint.completed_take_ids,
        saved
            .takes
            .iter()
            .map(|take| take.take_id.clone())
            .collect::<Vec<_>>()
    );
    assert!(!checkpoint.probe_completed && !checkpoint.bass_anchor_completed);
    for take in saved.takes {
        assert!(capture.join(take.wav_file).is_file());
    }
    let output = root.join("processed");
    assert!(crate::capture_session::clock::io::process_capture_session(&capture, &output).is_err());
    assert!(!output.exists());
}

#[test]
fn crash_during_probe_retains_a_nonterminal_checkpoint() {
    const CHILD_ROOT: &str = "SOTF_CHECKPOINT_CRASH_ROOT";
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        let root = std::path::PathBuf::from(root);
        let plan: CaptureSessionPlan =
            serde_json::from_slice(&std::fs::read(root.join("plan.json")).unwrap()).unwrap();
        let session = plan.validate().unwrap();
        let manifest = sweeps(&root, &session);
        run_auxiliary_passes(
            &session,
            &root.join("capture"),
            &Arc::new(AtomicBool::new(false)),
            manifest,
            |_, _, _, _, _, _, _, _, _, _, _| std::process::exit(74),
            |_, _, _, _, _, _, _, _, _, _, _, _, _, _, _| panic!("anchor must not run"),
        )
        .unwrap();
        panic!("crash point was not reached");
    }
    let (root, session) = prepare();
    std::fs::write(
        root.path().join("plan.json"),
        serde_json::to_vec(session.plan()).unwrap(),
    )
    .unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("capture_session::record::tests::checkpoint_tests::crash_during_probe_retains_a_nonterminal_checkpoint")
        .env(CHILD_ROOT, root.path())
        .stdout(std::process::Stdio::null())
        .status().unwrap();
    assert_eq!(status.code(), Some(74));
    assert_interrupted(root.path(), &session);
}

#[test]
fn probe_result_save_failure_cannot_commit_completion() {
    let (root, session) = prepare();
    let manifest = sweeps(root.path(), &session);
    let output = root.path().join("capture");
    let target = output.join("capture-raw.json");
    let backup = output.join("previous-journal.json");
    let error = run_auxiliary_passes(
        &session,
        &output,
        &Arc::new(AtomicBool::new(false)),
        manifest,
        |_, _, rate, _, _, _, _, _, wav, _, _| {
            // Deterministic cross-platform publication fault. Keep the previous
            // journal bytes in a backup while a directory blocks atomic rename.
            std::fs::rename(&target, &backup).unwrap();
            std::fs::create_dir(&target).unwrap();
            crate::signal_recorder::write_wav_file(wav, &[0.0; 8], rate, 1).unwrap();
            Ok(canned_probe())
        },
        |_, _, _, _, _, _, _, _, _, _, _, _, _, _, _| panic!("anchor must not run"),
    )
    .unwrap_err();
    assert!(error.contains("cannot save capture journal"), "{error}");
    std::fs::remove_dir(&target).unwrap();
    std::fs::rename(&backup, &target).unwrap();
    assert_interrupted(root.path(), &session);
}
