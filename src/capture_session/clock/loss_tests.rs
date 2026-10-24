//! Portable failure evidence through the real processing and consumer boundary.

// Rust guideline compliant 2026-02-21

use super::*;
use std::path::Path;

fn journal(root: &Path) -> RawCaptureManifest {
    serde_json::from_slice(&std::fs::read(root.join("capture-raw.json")).unwrap()).unwrap()
}

fn save(root: &Path, manifest: &RawCaptureManifest) -> Vec<u8> {
    let bytes = serde_json::to_vec_pretty(manifest).unwrap();
    std::fs::write(root.join("capture-raw.json"), &bytes).unwrap();
    bytes
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "Keep the full failed/cancelled acquisition, move, consumer load, and tamper refusal assertions together."
)]
fn selected_failed_or_cancelled_capture_retains_loss_in_a_moved_handoff() {
    for status in [RawCaptureStatus::Failed, RawCaptureStatus::Cancelled] {
        let root = write_golden_directory();
        let mut manifest = journal(root.path());
        manifest.plan.repeat_count = 2;
        manifest.status = status;
        manifest.checkpoint = Some(crate::capture_session::record::CaptureCheckpoint {
            phase: crate::capture_session::record::CapturePhase::Sweep {
                source_id: manifest.plan.sources[0].id.clone(),
                repeat_index: 1,
            },
            completed_take_ids: manifest
                .takes
                .iter()
                .map(|take| take.take_id.clone())
                .collect(),
            completed_loopbacks: Vec::new(),
            probe_completed: false,
            bass_anchor_completed: false,
        });
        let snapshot = crate::capture_session::loss_evidence::fixture(
            root.path(),
            "capture-loss-acquisition.json",
        );
        let original = std::fs::read(root.path().join(&snapshot.file)).unwrap();
        manifest.transport_evidence = vec![snapshot.clone()];
        manifest.error = Some(format!(
            "second repeat failed; acquisition evidence: {}",
            snapshot.file
        ));
        let raw_bytes = save(root.path(), &manifest);
        let output = root.path().join("processed");
        let report = io::process_capture_session_with_options(
            root.path(),
            &output,
            &io::ProcessOptions {
                selected_take_ids: manifest
                    .takes
                    .iter()
                    .map(|take| take.take_id.clone())
                    .collect(),
                ir_microphone: None,
            },
        )
        .unwrap();
        let handoff_file = autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME;
        let handoff_bytes = std::fs::read(output.join(handoff_file)).unwrap();
        let handoff: autoeq::capture_handoff::CaptureHandoff =
            serde_json::from_slice(&handoff_bytes).unwrap();
        handoff.validate().unwrap();
        assert_eq!(
            handoff.completion,
            if status == RawCaptureStatus::Failed {
                autoeq::capture_handoff::CaptureCompletion::Failed
            } else {
                autoeq::capture_handoff::CaptureCompletion::Cancelled
            }
        );
        let identity = handoff
            .artifacts
            .iter()
            .find(|asset| asset.file == snapshot.file)
            .unwrap();
        assert_eq!(
            identity.role,
            autoeq::capture_handoff::CaptureArtifactRole::SupportingEvidence
        );
        assert_eq!(identity.sha256, snapshot.sha256);
        assert_eq!(identity.bytes, u64::try_from(original.len()).unwrap());
        let moved_root = tempfile::tempdir().unwrap();
        let moved = moved_root.path().join("portable");
        std::fs::rename(&output, &moved).unwrap();
        drop(root); // No original report, journal, calibration, or audio is available.
        assert_eq!(std::fs::read(moved.join(&snapshot.file)).unwrap(), original);
        assert_eq!(
            std::fs::read(moved.join("capture-raw.json")).unwrap(),
            raw_bytes
        );
        autoeq::roomeq::load_config(&moved.join("recordings.json"), None).unwrap();
        assert!(
            crate::capture_session::handoff::publish_capture_handoff(
                &report,
                RawCaptureStatus::RawComplete,
                &raw_bytes,
                &moved
            )
            .unwrap_err()
            .contains("parent acquisition status")
        );
        std::fs::write(moved.join(&snapshot.file), b"changed diagnostic").unwrap();
        assert!(autoeq::roomeq::load_config(&moved.join("recordings.json"), None).is_err());
        assert!(
            crate::capture_session::handoff::publish_capture_handoff(
                &report, status, &raw_bytes, &moved
            )
            .unwrap_err()
            .contains("transport evidence changed")
        );
        assert_eq!(
            std::fs::read(moved.join(handoff_file)).unwrap(),
            handoff_bytes
        );
    }
}

#[test]
fn invalid_declared_loss_reports_fail_before_output_creation() {
    for mode in ["missing", "changed", "traversal", "malformed"] {
        let root = write_golden_directory();
        let mut manifest = journal(root.path());
        let mut snapshot = crate::capture_session::loss_evidence::fixture(
            root.path(),
            "capture-loss-acquisition.json",
        );
        match mode {
            "missing" => std::fs::remove_file(root.path().join(&snapshot.file)).unwrap(),
            "changed" => std::fs::write(root.path().join(&snapshot.file), b"changed").unwrap(),
            "traversal" => snapshot.file = "../capture-loss-acquisition.json".into(),
            "malformed" => {
                let bytes = b"{}";
                std::fs::write(root.path().join(&snapshot.file), bytes).unwrap();
                snapshot.sha256 = crate::capture_session::calibration_sha256(bytes);
            }
            _ => unreachable!(),
        }
        manifest.transport_evidence = vec![snapshot];
        save(root.path(), &manifest);
        let output = root.path().join("processed");
        assert!(
            io::process_capture_session(root.path(), &output).is_err(),
            "{mode}"
        );
        assert!(!output.exists(), "{mode}");
    }
}

#[test]
fn importing_legacy_journals_never_discovers_unreferenced_reports() {
    let root = write_golden_directory();
    let original = std::fs::read(root.path().join("capture-raw.json")).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&original).unwrap();
    assert!(value.get("transport_evidence").is_none());
    std::fs::write(
        root.path().join("capture-loss-unrelated.json"),
        b"private unrelated note",
    )
    .unwrap();
    let output = root.path().join("processed");
    io::process_capture_session(root.path(), &output).unwrap();
    assert!(!output.join("capture-loss-unrelated.json").exists());
    let handoff: autoeq::capture_handoff::CaptureHandoff = serde_json::from_slice(
        &std::fs::read(output.join(autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME)).unwrap(),
    )
    .unwrap();
    assert!(
        !handoff
            .artifacts
            .iter()
            .any(|asset| asset.file == "capture-loss-unrelated.json")
    );
}

#[test]
fn inconsistent_checkpoints_are_refused_before_output_creation() {
    use crate::capture_session::record::{CaptureCheckpoint, CapturePhase};
    for mode in ["bitmap", "phase", "undeclared", "valid"] {
        let root = write_golden_directory();
        let mut manifest = journal(root.path());
        let mut checkpoint = CaptureCheckpoint {
            phase: CapturePhase::Complete,
            completed_take_ids: manifest
                .takes
                .iter()
                .map(|take| take.take_id.clone())
                .collect(),
            completed_loopbacks: Vec::new(),
            probe_completed: false,
            bass_anchor_completed: false,
        };
        match mode {
            "bitmap" => {
                checkpoint.completed_take_ids.pop();
            }
            "phase" => checkpoint.phase = CapturePhase::Waking,
            "undeclared" => {
                checkpoint.phase = CapturePhase::Probe;
                manifest.status = RawCaptureStatus::Failed;
            }
            _ => {}
        }
        manifest.checkpoint = Some(checkpoint);
        save(root.path(), &manifest);
        let output = root.path().join("processed");
        if mode == "valid" {
            io::process_capture_session(root.path(), &output).unwrap();
        } else {
            assert!(
                io::process_capture_session(root.path(), &output)
                    .unwrap_err()
                    .contains("checkpoint"),
                "{mode}"
            );
            assert!(!output.exists());
        }
    }
}

#[test]
fn completed_capture_requires_its_declared_loopback_passes() {
    let root = write_golden_directory();
    let mut manifest = journal(root.path());
    manifest.plan.loopback = Some(crate::capture_session::CaptureLoopback {
        device: manifest.plan.microphones[0].device.clone(),
        input_channel: 7,
    });
    save(root.path(), &manifest);
    let output = root.path().join("processed");
    assert!(
        io::process_capture_session(root.path(), &output)
            .unwrap_err()
            .contains("missing declared loopback")
    );
    assert!(!output.exists());
}
