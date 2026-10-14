// Rust guideline compliant 2026-02-21

//! CTC raw-sweep assembly and head-position merging.
//!
//! Each CTC session records one head position: an ear-microphone pair plus a
//! loopback take per source, all through the standard chirp protocol.
//! [`assemble_ctc_takes`] crops the sweep window (chirps excluded) out of the
//! raw takes and writes stereo ear pairs, per-source loopbacks, a reference
//! sweep, and a `ctc-measurements.json` fragment. [`merge_ctc_fragments`]
//! combines fragments from repeated sessions into one self-contained bundle
//! whose `ctc.json` drops directly under `RoomEQ`'s `ctc:` key with
//! `matrix_source: "raw_sweep"`.
//!
//! Assembly deliberately uses raw takes, not clock-corrected ones: `RoomEQ`
//! aligns each take to its own loopback peak, so ears and loopback must share
//! one timeline. Drift across a single sweep is second order; prefer one
//! interface for ears and loopback so all three share a clock.

use autoeq::roomeq::{CtcHeadPositionConfig, CtcMeasurementConfig, CtcMeasurementFileConfig};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Per-position fragment written beside each processed CTC session.
pub const CTC_FRAGMENT_FILE: &str = "ctc-measurements.json";
/// Reference sweep WAV inside CTC bundles (mono, first channel significant).
pub const CTC_REFERENCE_FILE: &str = "ctc_reference_sweep.wav";
/// Merged CTC bundle descriptor, shaped as a partial `RoomEQ` `ctc:` block.
pub const CTC_MERGED_FILE: &str = "ctc.json";
/// Fragment schema version.
const CTC_FRAGMENT_VERSION: u32 = 1;
/// Ear roles backing the stereo channel order (1 = left, 2 = right).
const CTC_MIC_ROLES: [&str; 2] = ["left_ear", "right_ear"];

/// Assembled CTC files for one processed session.
#[derive(Debug)]
pub struct CtcAssembly {
    /// Fragment filename (`ctc-measurements.json`).
    pub fragment_file: String,
    /// Every CTC file written, fragment included, for handoff inventory.
    pub files: Vec<String>,
}

/// One head position's fragment, as written by [`assemble_ctc_takes`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CtcFragment {
    /// Fragment schema version (always [`CTC_FRAGMENT_VERSION`]).
    pub version: u32,
    /// Measurements block for this head position only.
    pub measurements: CtcMeasurementConfig,
    /// Reference sweep filename inside the session directory.
    pub reference_sweep: PathBuf,
    /// Played sweep seconds (journaled layout, not nominal).
    pub sweep_duration_s: f64,
    /// Sweep band edges in Hz.
    pub sweep_start_hz: f64,
    /// Sweep band edges in Hz.
    pub sweep_end_hz: f64,
    /// Session sample rate in Hz.
    pub sample_rate_hz: u32,
    /// SHA-256 of the reference sweep WAV bytes.
    pub reference_sha256: String,
}

/// Merged bundle summary for CLI reporting.
#[derive(Debug)]
pub struct MergedCtc {
    /// Head identities in merge order.
    pub heads: Vec<String>,
    /// Speaker identities.
    pub speakers: Vec<String>,
    /// Every file in the bundle, descriptor included.
    pub files: Vec<String>,
}

/// Crop raw takes into CTC raw-sweep files for one head position.
///
/// Returns `None` when the plan declares no CTC or when no selected matrix
/// exists (diagnostics-only runs assemble nothing, like the recording
/// projection). Both ears of a source must come from one repeat; anything
/// else is refused rather than silently mixed.
///
/// # Errors
/// Returns an error for missing or inconsistent takes, unsafe artifact paths, invalid WAVs, or output I/O failures.
#[expect(
    clippy::too_many_lines,
    reason = "Keep the ordered acquisition and artifact-processing phases together so cleanup and provenance remain explicit."
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Sweep sample counts are converted to f64 seconds, well within f64 integer precision."
)]
pub fn assemble_ctc_takes(
    report: &super::clock::io::ClockProcessedManifest,
    raw_root: &Path,
    output: &Path,
) -> Result<Option<CtcAssembly>, String> {
    let Some(ctc) = report.plan.ctc.as_ref() else {
        return Ok(None);
    };
    if report.selected_take_ids.is_none() {
        return Ok(None);
    }
    let selected = super::manifest::selected_takes(report)?;
    let layout = &report.stimulus;
    let window = layout.sweep_offset..layout.end_chirp_offset;
    let head_file = sanitize_filename_component(&ctc.head_position.id)?;
    let mut files = Vec::new();
    let mut used_names = BTreeSet::new();
    let mut reserve = |name: String| -> Result<String, String> {
        if !autoeq::capture_handoff::portable_capture_filename(&name)
            || !used_names.insert(name.clone())
        {
            return Err(format!("ctc output filename is unsafe or collides: {name}"));
        }
        files.push(name.clone());
        Ok(name)
    };
    let stimulus = read_mono_f32(
        &raw_artifact(raw_root, "stimulus.wav")?,
        report.plan.sample_rate_hz,
    )?;
    let sweep_end = layout.sweep_offset + layout.sweep_samples;
    let reference = stimulus
        .get(layout.sweep_offset..sweep_end)
        .ok_or("stimulus is shorter than the journaled sweep window")?;
    let reference_file = reserve(CTC_REFERENCE_FILE.into())?;
    crate::signal_recorder::write_wav_file(
        &output.join(&reference_file),
        reference,
        report.plan.sample_rate_hz,
        1,
    )?;
    let reference_sha256 = crate::capture_session::calibration_sha256(
        &std::fs::read(output.join(&reference_file)).map_err(|e| e.to_string())?,
    );
    let mut entries = Vec::new();
    for source in &report.plan.sources {
        let left = selected
            .get(&(source.id.as_str(), ctc.left_ear.as_str()))
            .ok_or_else(|| format!("ctc is missing the left-ear take for '{}'", source.id))?;
        let right = selected
            .get(&(source.id.as_str(), ctc.right_ear.as_str()))
            .ok_or_else(|| format!("ctc is missing the right-ear take for '{}'", source.id))?;
        if left.raw.repeat_index != right.raw.repeat_index {
            return Err(format!(
                "ctc ears for '{}' come from different repeats ({} vs {})",
                source.id, left.raw.repeat_index, right.raw.repeat_index
            ));
        }
        let loopback = report
            .loopback_files
            .iter()
            .find(|take| take.source_id == source.id && take.repeat_index == left.raw.repeat_index)
            .ok_or_else(|| {
                format!(
                    "ctc is missing the loopback take for '{}' repeat {}",
                    source.id, left.raw.repeat_index
                )
            })?;
        let ears = read_mono_f32(
            &raw_artifact(raw_root, &left.raw.wav_file)?,
            report.plan.sample_rate_hz,
        )?;
        let right_ear = read_mono_f32(
            &raw_artifact(raw_root, &right.raw.wav_file)?,
            report.plan.sample_rate_hz,
        )?;
        let loopback_samples = read_mono_f32(
            &raw_artifact(raw_root, &loopback.wav_file)?,
            report.plan.sample_rate_hz,
        )?;
        let ears = ears.get(window.clone()).ok_or_else(|| {
            format!(
                "left-ear take for '{}' is shorter than the sweep window",
                source.id
            )
        })?;
        let right_ear = right_ear.get(window.clone()).ok_or_else(|| {
            format!(
                "right-ear take for '{}' is shorter than the sweep window",
                source.id
            )
        })?;
        let loopback_samples = loopback_samples.get(window.clone()).ok_or_else(|| {
            format!(
                "loopback take for '{}' is shorter than the sweep window",
                source.id
            )
        })?;
        let source_file = sanitize_filename_component(&source.id)?;
        let ears_file = reserve(format!("ctc_{head_file}_{source_file}_ears.wav"))?;
        let mut stereo = Vec::with_capacity(ears.len() * 2);
        for (left_sample, right_sample) in ears.iter().zip(right_ear) {
            stereo.push(*left_sample);
            stereo.push(*right_sample);
        }
        crate::signal_recorder::write_wav_file(
            &output.join(&ears_file),
            &stereo,
            report.plan.sample_rate_hz,
            2,
        )?;
        let loopback_file = reserve(format!("ctc_{head_file}_{source_file}_loopback.wav"))?;
        crate::signal_recorder::write_wav_file(
            &output.join(&loopback_file),
            loopback_samples,
            report.plan.sample_rate_hz,
            1,
        )?;
        entries.push(CtcMeasurementFileConfig {
            head_position: ctc.head_position.id.clone(),
            speaker: source.id.clone(),
            ir: None,
            raw_sweep: Some(PathBuf::from(&ears_file)),
            loopback: Some(PathBuf::from(&loopback_file)),
        });
    }
    let fragment = CtcFragment {
        version: CTC_FRAGMENT_VERSION,
        measurements: CtcMeasurementConfig {
            speakers: report
                .plan
                .sources
                .iter()
                .map(|source| source.id.clone())
                .collect(),
            mics: CTC_MIC_ROLES
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            head_positions: vec![CtcHeadPositionConfig {
                id: ctc.head_position.id.clone(),
                x: ctc.head_position.x,
                y: ctc.head_position.y,
                z: ctc.head_position.z,
                yaw_deg: ctc.head_position.yaw_deg,
            }],
            files: entries,
        },
        reference_sweep: PathBuf::from(&reference_file),
        sweep_duration_s: layout.sweep_samples as f64 / f64::from(report.plan.sample_rate_hz),
        sweep_start_hz: report.plan.sweep.start_hz,
        sweep_end_hz: report.plan.sweep.end_hz,
        sample_rate_hz: report.plan.sample_rate_hz,
        reference_sha256,
    };
    let fragment_file = reserve(CTC_FRAGMENT_FILE.into())?;
    std::fs::write(
        output.join(&fragment_file),
        serde_json::to_vec_pretty(&fragment).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(Some(CtcAssembly {
        fragment_file: fragment_file.clone(),
        files,
    }))
}

/// Merge per-position fragments into one self-contained CTC bundle.
///
/// Every fragment must share speakers, sample rate, sweep parameters, and
/// reference bytes; head identities must be distinct; every head/speaker
/// cell must be present. Anything else is refused. `output` must not exist.
///
/// # Errors
/// Returns an error if fragments disagree, identities or files are invalid, or bundle I/O fails.
#[expect(
    clippy::too_many_lines,
    reason = "Keep the ordered acquisition and artifact-processing phases together so cleanup and provenance remain explicit."
)]
#[expect(
    clippy::float_cmp,
    reason = "Serialized sweep settings must match exactly before fragments can share one reference waveform."
)]
pub fn merge_ctc_fragments(directories: &[PathBuf], output: &Path) -> Result<MergedCtc, String> {
    if directories.is_empty() {
        return Err("ctc merge requires at least one session directory".into());
    }
    if output.exists() {
        return Err(format!(
            "refusing to overwrite existing {}",
            output.display()
        ));
    }
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let mut fragments = Vec::new();
    for directory in directories {
        let bytes = std::fs::read(directory.join(CTC_FRAGMENT_FILE)).map_err(|e| {
            format!(
                "cannot read {} in {}: {e}",
                CTC_FRAGMENT_FILE,
                directory.display()
            )
        })?;
        let fragment: CtcFragment =
            serde_json::from_slice(&bytes).map_err(|e| format!("invalid ctc fragment: {e}"))?;
        if fragment.version != CTC_FRAGMENT_VERSION {
            return Err(format!(
                "unsupported ctc fragment version {}",
                fragment.version
            ));
        }
        fragments.push((directory, fragment));
    }
    let (_, first) = &fragments[0];
    let mut heads = Vec::new();
    let mut seen_heads = BTreeSet::new();
    let mut measurements = CtcMeasurementConfig {
        speakers: first.measurements.speakers.clone(),
        mics: first.measurements.mics.clone(),
        head_positions: Vec::new(),
        files: Vec::new(),
    };
    for (directory, fragment) in &fragments {
        if fragment.measurements.speakers != measurements.speakers
            || fragment.measurements.mics != measurements.mics
            || fragment.sample_rate_hz != first.sample_rate_hz
            || fragment.sweep_duration_s != first.sweep_duration_s
            || fragment.sweep_start_hz != first.sweep_start_hz
            || fragment.sweep_end_hz != first.sweep_end_hz
            || fragment.reference_sha256 != first.reference_sha256
        {
            return Err(format!(
                "ctc fragment in {} disagrees on speakers, rate, sweep, or reference",
                directory.display()
            ));
        }
        let reference_bytes = std::fs::read(directory.join(&fragment.reference_sweep))
            .map_err(|e| format!("cannot read ctc reference sweep: {e}"))?;
        if crate::capture_session::calibration_sha256(&reference_bytes) != fragment.reference_sha256
        {
            return Err(format!(
                "ctc reference sweep in {} does not match its fragment hash",
                directory.display()
            ));
        }
        if fragment.measurements.head_positions.len() != 1 {
            return Err(format!(
                "ctc fragment in {} must carry exactly one head position",
                directory.display()
            ));
        }
        let head = &fragment.measurements.head_positions[0];
        if !seen_heads.insert(head.id.clone()) {
            return Err(format!("duplicate ctc head position '{}'", head.id));
        }
        heads.push(head.id.clone());
        measurements.head_positions.push(head.clone());
        measurements
            .files
            .extend(fragment.measurements.files.clone());
    }
    for head in &heads {
        for speaker in &measurements.speakers {
            if !measurements
                .files
                .iter()
                .any(|entry| &entry.head_position == head && &entry.speaker == speaker)
            {
                return Err(format!(
                    "ctc matrix is missing head '{head}' speaker '{speaker}'"
                ));
            }
        }
    }
    let mut bundled = Vec::new();
    let mut copy = |directory: &Path, name: &Path| -> Result<(), String> {
        let file = name
            .file_name()
            .and_then(|stem| stem.to_str())
            .ok_or("ctc bundle member must be a local filename")?;
        if !autoeq::capture_handoff::portable_capture_filename(file) {
            return Err(format!("ctc bundle member is not portable: {file}"));
        }
        let bytes = std::fs::read(directory.join(name))
            .map_err(|e| format!("cannot read ctc file: {e}"))?;
        let target = output.join(file);
        if target.exists() {
            let kept = std::fs::read(&target).map_err(|e| e.to_string())?;
            if kept != bytes {
                return Err(format!("ctc bundle filename collides: {file}"));
            }
            return Ok(());
        }
        std::fs::write(&target, bytes).map_err(|e| e.to_string())?;
        bundled.push(file.to_string());
        Ok(())
    };
    for (directory, fragment) in &fragments {
        for entry in &fragment.measurements.files {
            let raw = entry
                .raw_sweep
                .as_deref()
                .ok_or("ctc merge requires raw_sweep files")?;
            let loopback = entry
                .loopback
                .as_deref()
                .ok_or("ctc merge requires loopback files")?;
            copy(directory, raw)?;
            copy(directory, loopback)?;
        }
    }
    copy(&directories[0], &first.reference_sweep)?;
    let reference_name = first
        .reference_sweep
        .file_name()
        .and_then(|stem| stem.to_str())
        .ok_or("ctc reference sweep must be a local filename")?;
    for entry in &mut measurements.files {
        for path in [&mut entry.raw_sweep, &mut entry.loopback]
            .into_iter()
            .flatten()
        {
            *path = PathBuf::from(
                path.file_name()
                    .and_then(|stem| stem.to_str())
                    .ok_or("ctc bundle member must be a local filename")?,
            );
        }
    }
    let merged = serde_json::json!({
        "enabled": true,
        "matrix_source": "raw_sweep",
        "measurements": measurements,
        "reference_sweep": reference_name,
        "sweep_duration_s": first.sweep_duration_s,
        "sweep_start_hz": first.sweep_start_hz,
        "sweep_end_hz": first.sweep_end_hz,
    });
    std::fs::write(
        output.join(CTC_MERGED_FILE),
        serde_json::to_vec_pretty(&merged).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    bundled.push(CTC_MERGED_FILE.into());
    Ok(MergedCtc {
        heads,
        speakers: measurements.speakers.clone(),
        files: bundled,
    })
}

/// Reduce a user label to a portable filename component.
fn sanitize_filename_component(label: &str) -> Result<String, String> {
    let clean: String = label
        .chars()
        .map(|cell| {
            if cell.is_ascii_alphanumeric() || cell == '_' || cell == '-' || cell == '.' {
                cell
            } else {
                '_'
            }
        })
        .collect();
    if clean.is_empty() || clean.chars().all(|cell| cell == '.') {
        return Err(format!("label has no portable filename form: {label:?}"));
    }
    Ok(clean)
}

/// Resolve a journaled raw artifact without following user traversal.
fn raw_artifact(raw_root: &Path, file: &str) -> Result<PathBuf, String> {
    if !autoeq::capture_handoff::portable_capture_filename(file) {
        return Err(format!("raw artifact path is not portable: {file}"));
    }
    Ok(raw_root.join(file))
}

/// Read a mono 32-bit float WAV fully.
fn read_mono_f32(path: &Path, sample_rate_hz: u32) -> Result<Vec<f32>, String> {
    let mut reader =
        hound::WavReader::open(path).map_err(|e| format!("cannot read capture WAV: {e}"))?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.sample_rate != sample_rate_hz
        || spec.bits_per_sample != 32
        || spec.sample_format != hound::SampleFormat::Float
    {
        return Err(format!(
            "ctc requires mono f32 WAVs at the session rate: {}",
            path.display()
        ));
    }
    reader
        .samples::<f32>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("cannot decode capture WAV: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizer_neutralizes_traversal_and_keeps_plain_labels() {
        assert_eq!(sanitize_filename_component("head-0").unwrap(), "head-0");
        assert_eq!(sanitize_filename_component("seat A/B").unwrap(), "seat_A_B");
        assert_eq!(sanitize_filename_component("../x").unwrap(), ".._x");
        assert!(sanitize_filename_component("").is_err());
        assert!(sanitize_filename_component("...").is_err());
    }

    fn fragment(dir: &Path, head: &str, speakers: &[&str]) -> CtcFragment {
        let files: Vec<CtcMeasurementFileConfig> = speakers
            .iter()
            .map(|speaker| CtcMeasurementFileConfig {
                head_position: head.into(),
                speaker: (*speaker).into(),
                ir: None,
                raw_sweep: Some(PathBuf::from(format!("ctc_{head}_{speaker}_ears.wav"))),
                loopback: Some(PathBuf::from(format!("ctc_{head}_{speaker}_loopback.wav"))),
            })
            .collect();
        for entry in &files {
            for member in [&entry.raw_sweep, &entry.loopback] {
                std::fs::write(dir.join(member.as_deref().unwrap()), b"wav-bytes").unwrap();
            }
        }
        std::fs::write(dir.join(CTC_REFERENCE_FILE), b"reference-bytes").unwrap();
        CtcFragment {
            version: CTC_FRAGMENT_VERSION,
            measurements: CtcMeasurementConfig {
                speakers: speakers
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect(),
                mics: CTC_MIC_ROLES
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect(),
                head_positions: vec![CtcHeadPositionConfig {
                    id: head.into(),
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    yaw_deg: 0.0,
                }],
                files,
            },
            reference_sweep: PathBuf::from(CTC_REFERENCE_FILE),
            sweep_duration_s: 5.0,
            sweep_start_hz: 20.0,
            sweep_end_hz: 20000.0,
            sample_rate_hz: 48000,
            reference_sha256: crate::capture_session::calibration_sha256(b"reference-bytes"),
        }
    }

    fn write_fragment(dir: &Path, fragment: &CtcFragment) {
        std::fs::write(
            dir.join(CTC_FRAGMENT_FILE),
            serde_json::to_vec_pretty(fragment).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn merge_combines_two_heads_into_a_self_contained_bundle() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        write_fragment(
            first.path(),
            &fragment(first.path(), "head-0", &["left", "right"]),
        );
        write_fragment(
            second.path(),
            &fragment(second.path(), "head-1", &["left", "right"]),
        );
        let output = second.path().join("merged");
        let merged = merge_ctc_fragments(
            &[first.path().to_path_buf(), second.path().to_path_buf()],
            &output,
        )
        .unwrap();
        assert_eq!(
            merged.heads,
            vec!["head-0".to_string(), "head-1".to_string()]
        );
        assert_eq!(
            merged.speakers,
            vec!["left".to_string(), "right".to_string()]
        );
        assert_eq!(merged.files.len(), 10);
        let bundle: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output.join(CTC_MERGED_FILE)).unwrap()).unwrap();
        assert_eq!(bundle["matrix_source"], "raw_sweep");
        assert_eq!(bundle["measurements"]["files"].as_array().unwrap().len(), 4);
        assert_eq!(
            bundle["measurements"]["head_positions"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        for file in &merged.files {
            assert!(output.join(file).is_file(), "{file}");
        }
    }

    #[test]
    fn merge_refuses_mismatches_gaps_and_tampering() {
        let root = tempfile::tempdir().unwrap();
        let good = fragment(root.path(), "head-0", &["left", "right"]);
        // Mismatched speakers.
        let other = root.path().join("other");
        std::fs::create_dir(&other).unwrap();
        let mut drifted = fragment(&other, "head-1", &["left", "center"]);
        write_fragment(&other, &drifted);
        write_fragment(root.path(), &good);
        assert!(
            merge_ctc_fragments(
                &[root.path().to_path_buf(), other.clone()],
                &root.path().join("merged-speakers")
            )
            .is_err()
        );
        // Duplicate head identity.
        drifted = fragment(&other, "head-0", &["left", "right"]);
        write_fragment(&other, &drifted);
        assert!(
            merge_ctc_fragments(
                &[root.path().to_path_buf(), other.clone()],
                &root.path().join("merged-heads")
            )
            .is_err()
        );
        // Missing matrix cell.
        let mut gapped = fragment(&other, "head-1", &["left", "right"]);
        gapped.measurements.files.pop();
        write_fragment(&other, &gapped);
        assert!(
            merge_ctc_fragments(
                &[root.path().to_path_buf(), other.clone()],
                &root.path().join("merged-gap")
            )
            .is_err()
        );
        // Tampered reference bytes.
        let tampered = root.path().join("tampered");
        std::fs::create_dir(&tampered).unwrap();
        write_fragment(
            &tampered,
            &fragment(&tampered, "head-2", &["left", "right"]),
        );
        std::fs::write(tampered.join(CTC_REFERENCE_FILE), b"other-bytes").unwrap();
        assert!(
            merge_ctc_fragments(
                &[root.path().to_path_buf(), tampered],
                &root.path().join("merged-tamper")
            )
            .is_err()
        );
    }
}
