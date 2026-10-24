//! Bounded, hash-bound acquisition diagnostics retained with portable captures.

// Rust guideline compliant 2026-02-21

use super::record::CaptureFileSnapshot;
use std::collections::HashSet;
use std::path::Path;

// Enough for bounded wake retries and acquisition failures, without unbounded
// metadata allocation when importing a supplied journal.
const MAX_REPORTS: usize = 512;
const MAX_REPORT_BYTES: usize = 1024 * 1024;
const MAX_TOTAL_BYTES: usize = 8 * 1024 * 1024;

#[derive(serde::Deserialize)]
struct LossReport {
    version: u32,
    error: String,
    inputs: Vec<crate::signal_recorder::CaptureInputEvidence>,
}

fn report_name(name: &str) -> Result<(), String> {
    crate::artifacts::member_name(Path::new(name))?;
    if !is_report(name) {
        return Err("transport evidence needs a capture-loss-*.json filename".into());
    }
    Ok(())
}

fn is_report(name: &str) -> bool {
    name.starts_with("capture-loss-")
        && Path::new(name)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
}

fn validate(bytes: &[u8]) -> Result<(), String> {
    let report: LossReport = serde_json::from_slice(bytes)
        .map_err(|e| format!("invalid transport evidence JSON: {e}"))?;
    if report.version != 1
        || report.error.is_empty()
        || report.inputs.is_empty()
        || report.inputs.len() > 64
    {
        return Err("invalid transport evidence header or input count".into());
    }
    for input in report.inputs {
        let clock = input.continuity;
        if input.device_id.is_empty()
            || input.routes.is_empty()
            || input.routes.len() > 64
            || clock.intervals.len() > 128
            || (clock.omitted_intervals > 0 && clock.intervals.len() != 128)
        {
            return Err("invalid transport evidence device, routes, or interval count".into());
        }
        let mut end = 0;
        let mut total = 0_usize;
        for interval in clock.intervals {
            if interval.start_frame < end
                || interval.end_frame <= interval.start_frame
                || interval.end_frame > clock.observed_frames
            {
                return Err("invalid transport evidence interval positions".into());
            }
            total = total
                .checked_add(interval.end_frame - interval.start_frame)
                .ok_or("transport evidence count overflow")?;
            end = interval.end_frame;
        }
        if total > clock.missing_frames_per_channel
            || clock.missing_frames_per_channel > clock.observed_frames
            || clock.omitted_intervals > clock.missing_frames_per_channel.saturating_sub(total)
            || (clock.omitted_intervals == 0 && total != clock.missing_frames_per_channel)
        {
            return Err(
                "transport evidence intervals disagree with the missing-frame count".into(),
            );
        }
    }
    Ok(())
}

/// Read only explicitly declared reports when importing a saved capture.
pub(super) fn read_verified(
    root: &Path,
    reports: &[CaptureFileSnapshot],
) -> Result<Vec<(String, Vec<u8>)>, String> {
    if reports.len() > MAX_REPORTS {
        return Err("capture exceeds the transport evidence report limit".into());
    }
    let mut names = HashSet::new();
    let mut total = 0_usize;
    let mut frozen = Vec::with_capacity(reports.len());
    for report in reports {
        report_name(&report.file)?;
        if !names.insert(report.file.to_ascii_lowercase()) {
            return Err("transport evidence filename is duplicated".into());
        }
        let bytes = crate::artifacts::read_member(root, Path::new(&report.file), MAX_REPORT_BYTES)?;
        total = total
            .checked_add(bytes.len())
            .ok_or("transport evidence size overflow")?;
        if total > MAX_TOTAL_BYTES {
            return Err("capture exceeds the aggregate transport evidence byte limit".into());
        }
        if super::calibration_sha256(&bytes) != report.sha256 {
            return Err(format!("transport evidence changed: {}", report.file));
        }
        validate(&bytes)?;
        frozen.push((report.file.clone(), bytes));
    }
    Ok(frozen)
}

/// Discover reports only while writing this process's newly created capture.
/// Imported sessions never discover artifacts from directory contents.
#[cfg(any(test, feature = "capture-io"))]
pub(super) fn refresh(root: &Path, reports: &mut Vec<CaptureFileSnapshot>) -> Result<(), String> {
    let existing = read_verified(root, reports)?;
    let mut total: usize = existing.iter().map(|(_, bytes)| bytes.len()).sum();
    let mut known: std::collections::HashMap<_, _> = reports
        .iter()
        .map(|report| (report.file.to_ascii_lowercase(), report.file.clone()))
        .collect();
    let mut updated = reports.clone();
    for entry in std::fs::read_dir(root).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !is_report(&name) {
            continue;
        }
        if let Some(existing) = known.get(&name.to_ascii_lowercase()) {
            if existing != &name {
                return Err("transport evidence filenames collide after case folding".into());
            }
            continue;
        }
        report_name(&name)?;
        if updated.len() == MAX_REPORTS {
            return Err("capture exceeds the transport evidence report limit".into());
        }
        let bytes = crate::artifacts::read_member(root, Path::new(&name), MAX_REPORT_BYTES)?;
        total = total
            .checked_add(bytes.len())
            .ok_or("transport evidence size overflow")?;
        if total > MAX_TOTAL_BYTES {
            return Err("capture exceeds the aggregate transport evidence byte limit".into());
        }
        validate(&bytes)?;
        known.insert(name.to_ascii_lowercase(), name.clone());
        updated.push(CaptureFileSnapshot {
            file: name,
            sha256: super::calibration_sha256(&bytes),
        });
    }
    updated.sort_unstable_by(|left, right| left.file.cmp(&right.file));
    *reports = updated;
    Ok(())
}

#[cfg(test)]
pub(super) fn fixture(root: &Path, name: &str) -> CaptureFileSnapshot {
    let bytes = serde_json::to_vec(&serde_json::json!({
        "version":1, "error":"synthetic input ring overrun", "capture_kind":"test",
        "timeline_origin":"first input callback", "output_device_id":"synthetic-output",
        "output_device_selector":null, "take_index":null, "output_channels":[0],
        "output_channel":0, "stimulus_frames":24,
        "inputs":[{"clock_index":0,"device_selector":"synthetic input","device_id":"synthetic-native-id",
            "sample_rate_hz":48000,"routes":[{"input_index":0,"hardware_channel":3}],
            "continuity":{"observed_frames":32,"missing_frames_per_channel":3,
                "invalid_block_or_stream_error":false,"omitted_intervals":0,
                "intervals":[{"start_frame":12,"end_frame":15}]}}]
    })).unwrap();
    std::fs::write(root.join(name), &bytes).unwrap();
    CaptureFileSnapshot {
        file: name.into(),
        sha256: super::calibration_sha256(&bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_ignores_unrelated_files_and_never_adopts_changed_evidence() {
        let root = tempfile::tempdir().unwrap();
        let snapshot = fixture(root.path(), "capture-loss-a.json");
        std::fs::write(root.path().join("notes.json"), b"private unrelated note").unwrap();
        std::fs::write(root.path().join(".capture-loss-partial.json"), b"partial").unwrap();
        let mut reports = Vec::new();
        refresh(root.path(), &mut reports).unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].sha256, snapshot.sha256);
        std::fs::write(root.path().join(&snapshot.file), b"changed").unwrap();
        assert!(
            refresh(root.path(), &mut reports)
                .unwrap_err()
                .contains("changed")
        );
        assert_eq!(reports[0].sha256, snapshot.sha256);
    }

    #[test]
    fn report_paths_counts_and_declared_hashes_are_enforced() {
        let root = tempfile::tempdir().unwrap();
        let snapshot = fixture(root.path(), "capture-loss-a.json");
        for name in [
            "../capture-loss-a.json",
            "/capture-loss-a.json",
            "nested/capture-loss-a.json",
            "stimulus.wav",
        ] {
            let mut report = snapshot.clone();
            report.file = name.into();
            assert!(read_verified(root.path(), &[report]).is_err(), "{name}");
        }
        let mut duplicate = snapshot.clone();
        duplicate.file = "capture-loss-A.json".into();
        assert!(
            read_verified(root.path(), &[snapshot.clone(), duplicate])
                .unwrap_err()
                .contains("duplicated")
        );
        assert!(
            read_verified(root.path(), &vec![snapshot.clone(); MAX_REPORTS + 1])
                .unwrap_err()
                .contains("report limit")
        );
        let mut wrong_hash = snapshot;
        wrong_hash.sha256 = "0".repeat(64);
        assert!(
            read_verified(root.path(), &[wrong_hash])
                .unwrap_err()
                .contains("changed")
        );
    }

    #[test]
    fn individual_and_aggregate_report_bytes_are_bounded() {
        let root = tempfile::tempdir().unwrap();
        let first = fixture(root.path(), "capture-loss-a.json");
        let mut bytes = std::fs::read(root.path().join(&first.file)).unwrap();
        bytes.resize(MAX_REPORT_BYTES + 1, b' ');
        std::fs::write(root.path().join(&first.file), &bytes).unwrap();
        assert!(
            read_verified(root.path(), &[first])
                .unwrap_err()
                .contains("size limit")
        );
        bytes.truncate(MAX_REPORT_BYTES);
        let reports: Vec<_> = (0..9)
            .map(|index| {
                let file = format!("capture-loss-{index}.json");
                std::fs::write(root.path().join(&file), &bytes).unwrap();
                CaptureFileSnapshot {
                    file,
                    sha256: super::super::calibration_sha256(&bytes),
                }
            })
            .collect();
        assert!(read_verified(root.path(), &reports[..8]).is_ok());
        assert!(
            read_verified(root.path(), &reports)
                .unwrap_err()
                .contains("aggregate")
        );
    }

    #[test]
    fn invalid_interval_evidence_is_rejected_even_with_a_matching_hash() {
        let root = tempfile::tempdir().unwrap();
        let snapshot = fixture(root.path(), "capture-loss-a.json");
        let original: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.path().join(&snapshot.file)).unwrap())
                .unwrap();
        for (field, value) in [
            ("observed_frames", 14),
            ("missing_frames_per_channel", 2),
            ("omitted_intervals", 1),
        ] {
            let mut report = original.clone();
            report["inputs"][0]["continuity"][field] = value.into();
            let bytes = serde_json::to_vec(&report).unwrap();
            std::fs::write(root.path().join(&snapshot.file), &bytes).unwrap();
            let reference = CaptureFileSnapshot {
                file: snapshot.file.clone(),
                sha256: super::super::calibration_sha256(&bytes),
            };
            assert!(read_verified(root.path(), &[reference]).is_err(), "{field}");
        }
    }

    #[test]
    fn truncated_reports_must_have_possible_missing_frame_totals() {
        let root = tempfile::tempdir().unwrap();
        let report = fixture(root.path(), "capture-loss-a.json");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.path().join(report.file)).unwrap()).unwrap();
        let clock = &mut value["inputs"][0]["continuity"];
        clock["intervals"] = (0..128)
            .map(|start| serde_json::json!({"start_frame":start,"end_frame":start+1}))
            .collect();
        clock["observed_frames"] = 256.into();
        clock["missing_frames_per_channel"] = 129.into();
        clock["omitted_intervals"] = 1.into();
        assert!(validate(&serde_json::to_vec(&value).unwrap()).is_ok());
        value["inputs"][0]["continuity"]["missing_frames_per_channel"] = 257.into();
        assert!(validate(&serde_json::to_vec(&value).unwrap()).is_err());
        value["inputs"][0]["continuity"]["missing_frames_per_channel"] = 129.into();
        value["inputs"][0]["continuity"]["omitted_intervals"] = 2.into();
        assert!(validate(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn report_symlinks_are_never_followed() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let report = fixture(outside.path(), "capture-loss-a.json");
        std::os::unix::fs::symlink(
            outside.path().join(&report.file),
            root.path().join(&report.file),
        )
        .unwrap();
        assert!(read_verified(root.path(), &[report]).is_err());
        assert!(refresh(root.path(), &mut Vec::new()).is_err());
    }
}
