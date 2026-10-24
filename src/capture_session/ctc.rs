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
//! Both ears and loopback must share one hardware input timeline. A single
//! electrical-loopback clock fit maps all three onto the stimulus timeline;
//! independent acoustic-peak alignment would erase physical ear delays.

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
            || output.join(&name).exists()
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
    let chirp = read_mono_f32(
        &raw_artifact(raw_root, "timing-chirp.wav")?,
        report.plan.sample_rate_hz,
    )?;
    // The electrical reference has no microphone travel correction. Reuse
    // the marker estimator with acoustic geometry explicitly disabled.
    let mut timing_plan = report.plan.clone();
    timing_plan.timing_reference = None;
    let timing_manifest = super::record::RawCaptureManifest {
        checkpoint: None,
        transport_evidence: Vec::new(),
        localization_calibration: None,
        version: 1,
        plan: timing_plan,
        stimulus: layout.clone(),
        timing_reference_output_channel: report.timing_reference_output_channel,
        status: super::record::RawCaptureStatus::RawComplete,
        pending_processing: Vec::new(),
        calibrations: Vec::new(),
        takes: Vec::new(),
        loopback_files: Vec::new(),
        error: None,
        probe_pass: None,
        bass_anchor_pass: None,
    };
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
        if left.raw.device_id.is_empty()
            || left.raw.device_id != right.raw.device_id
            || left.raw.device_id != loopback.device_id
            || left.raw.samples != right.raw.samples
            || left.raw.samples != loopback.samples
            || left.raw.input_sample_format != right.raw.input_sample_format
            || left.raw.input_sample_format != loopback.input_sample_format
            || left.raw.output_device_id != right.raw.output_device_id
        {
            return Err(
                "ctc ears and loopback must share one verified hardware input timeline".into(),
            );
        }
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
        if ears.len() != left.raw.samples
            || right_ear.len() != right.raw.samples
            || loopback_samples.len() != loopback.samples
        {
            return Err("ctc raw lengths disagree with the journal".into());
        }
        let mut electrical_take = left.raw.clone();
        electrical_take.microphone_id.clear();
        let (clock, fit) = super::clock::fit_capture_clock(
            &timing_manifest,
            &electrical_take,
            &loopback_samples,
            &chirp,
        );
        let fit = fit.ok_or_else(|| {
            format!("ctc loopback markers do not establish a common clock mapping: {clock:?}")
        })?;
        let map = |samples: &[f32]| -> Result<Vec<f32>, String> {
            let ratio = 1.0 + fit.skew.skew_ppm / 1e6;
            let first = fit.skew.offset_samples + window.start as f64 * ratio;
            let end = fit.skew.offset_samples + window.end as f64 * ratio;
            if !first.is_finite() || !end.is_finite() || first < 0.0 || end > samples.len() as f64 {
                return Err(
                    "ctc raw input does not contain the complete mapped sweep and decay".into(),
                );
            }
            math_audio_dsp::capture_resample::resample_to_common_clock(
                samples,
                &fit.skew,
                layout.total_samples,
            )
            .ok_or_else(|| "ctc common-clock resampling refused the mapping".into())
        };
        // Apply the identical mapping to all channels, preserving interaural
        // delay and speaker-to-ear travel relative to the electrical loopback.
        let ears = map(&ears)?;
        let right_ear = map(&right_ear)?;
        let loopback_samples = map(&loopback_samples)?;
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
        // Markers establish the common clock, but do not prove the electrical
        // monitor contains this source's sweep (routing can carry chirps only).
        let lag =
            math_audio_dsp::analysis::estimate_lag_with_confidence(reference, loopback_samples)
                .map_err(|error| format!("ctc loopback sweep is unusable: {error}"))?;
        let minimum =
            math_audio_dsp::analysis::MeasurementQualityConfig::default().minimum_lag_confidence;
        if lag.confidence < minimum
            || lag.lag_samples < 0
            || usize::try_from(lag.lag_samples)
                .ok()
                .and_then(|lag| lag.checked_add(reference.len()))
                .is_none_or(|end| end > loopback_samples.len())
        {
            return Err(
                "ctc loopback sweep is missing, truncated, or does not match the reference".into(),
            );
        }
        for samples in [ears, right_ear, loopback_samples] {
            if samples.iter().all(|sample| *sample == 0.0)
                || samples.iter().any(|sample| sample.abs() >= 1.0)
            {
                return Err("ctc sweep contains silent or digitally clipped input".into());
            }
        }
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
pub fn merge_ctc_fragments(directories: &[PathBuf], output: &Path) -> Result<MergedCtc, String> {
    // A failed import must not publish a partial or apparently usable bundle.
    if directories.is_empty() || directories.len() > 128 {
        return Err("ctc merge requires 1..=128 session directories".into());
    }
    let staging = crate::artifacts::StagedDirectory::new(output)?;
    let merged = merge_into(directories, staging.path())?;
    staging.publish()?;
    Ok(merged)
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep the ordered acquisition and artifact-processing phases together so cleanup and provenance remain explicit."
)]
#[expect(
    clippy::float_cmp,
    reason = "Serialized sweep settings must match exactly before fragments can share one reference waveform."
)]
fn merge_into(directories: &[PathBuf], output: &Path) -> Result<MergedCtc, String> {
    use crate::artifacts::{MAX_AUDIO_BYTES, MAX_METADATA_BYTES, read_member};
    let mut fragments = Vec::new();
    let mut metadata_bytes = 0_usize;
    for directory in directories {
        let bytes = read_member(directory, Path::new(CTC_FRAGMENT_FILE), MAX_METADATA_BYTES)
            .map_err(|e| {
                format!(
                    "cannot read {} in {}: {e}",
                    CTC_FRAGMENT_FILE,
                    directory.display()
                )
            })?;
        metadata_bytes = metadata_bytes
            .checked_add(bytes.len())
            .ok_or("ctc metadata size overflow")?;
        if metadata_bytes > 32 * 1024 * 1024 {
            return Err("ctc metadata exceeds the aggregate 32 MiB limit".into());
        }
        let fragment: CtcFragment =
            serde_json::from_slice(&bytes).map_err(|e| format!("invalid ctc fragment: {e}"))?;
        if fragment.version != CTC_FRAGMENT_VERSION {
            return Err(format!(
                "unsupported ctc fragment version {}",
                fragment.version
            ));
        }
        validate_fragment(&fragment)?;
        fragments.push((directory, fragment));
    }
    let (_, first) = &fragments[0];
    let mut heads = Vec::new();
    let mut seen_heads = BTreeSet::new();
    let mut read_bytes = 0_usize;
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
        if fragment.measurements.files.len() > 4096 {
            return Err("ctc fragment contains too many measurements".into());
        }
        let reference_bytes = read_member(directory, &fragment.reference_sweep, MAX_AUDIO_BYTES)
            .map_err(|e| format!("cannot read ctc reference sweep: {e}"))?;
        read_bytes = read_bytes
            .checked_add(reference_bytes.len())
            .ok_or("ctc import size overflow")?;
        if read_bytes > 1024 * 1024 * 1024 {
            return Err("ctc import exceeds its read budget".into());
        }
        let reference_frames = validate_ctc_wave(&reference_bytes, 1, fragment.sample_rate_hz)?;
        if (f64::from(reference_frames)
            - fragment.sweep_duration_s * f64::from(fragment.sample_rate_hz))
        .abs()
            > 1.0
        {
            return Err("ctc reference duration disagrees with its fragment".into());
        }
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
    // Limit the whole bundle as well as individual files; keep at most one
    // member buffer alive while copying potentially many fragments.
    let mut total_bytes = 0_usize;
    let mut copy =
        |directory: &Path, name: &Path, channels: u16, hash: Option<&str>| -> Result<u32, String> {
            let file = crate::artifacts::member_name(name)?;
            let bytes = read_member(directory, name, MAX_AUDIO_BYTES)
                .map_err(|e| format!("cannot read ctc file: {e}"))?;
            read_bytes = read_bytes
                .checked_add(bytes.len())
                .ok_or("ctc import size overflow")?;
            if read_bytes > 1024 * 1024 * 1024 {
                return Err("ctc import exceeds its read budget".into());
            }
            let frames = validate_ctc_wave(&bytes, channels, first.sample_rate_hz)?;
            if hash.is_some_and(|hash| crate::capture_session::calibration_sha256(&bytes) != hash) {
                return Err("ctc reference changed during merge".into());
            }
            let target = output.join(file);
            if target.exists() {
                let kept = std::fs::read(&target).map_err(|e| e.to_string())?;
                if kept != bytes {
                    return Err(format!("ctc bundle filename collides: {file}"));
                }
                return Ok(frames);
            }
            total_bytes = total_bytes
                .checked_add(bytes.len())
                .ok_or("ctc bundle size overflow")?;
            if total_bytes > 1024 * 1024 * 1024 {
                return Err("ctc bundle exceeds its 1 GiB size limit".into());
            }
            std::fs::write(&target, bytes).map_err(|e| e.to_string())?;
            bundled.push(file.to_string());
            Ok(frames)
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
            let ears = copy(directory, raw, 2, None)?;
            let loopback = copy(directory, loopback, 1, None)?;
            if ears != loopback
                || f64::from(ears) < first.sweep_duration_s * f64::from(first.sample_rate_hz)
            {
                return Err(
                    "ctc ear/loopback lengths differ or omit part of the reference sweep".into(),
                );
            }
        }
    }
    copy(
        &directories[0],
        &first.reference_sweep,
        1,
        Some(&first.reference_sha256),
    )?;
    let reference_name = first
        .reference_sweep
        .file_name()
        .and_then(|stem| stem.to_str())
        .ok_or("ctc reference sweep must be a local filename")?;
    for entry in &mut measurements.files {
        entry.ir = None;
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
    crate::artifacts::member_path(raw_root, Path::new(file))
}

fn validate_ctc_wave(bytes: &[u8], channels: u16, rate: u32) -> Result<u32, String> {
    let mut reader =
        hound::WavReader::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    if spec.channels != channels
        || spec.sample_rate != rate
        || spec.bits_per_sample != 32
        || spec.sample_format != hound::SampleFormat::Float
        || reader.duration() == 0
        || reader.duration() > 16_000_000
    {
        return Err("ctc WAV has invalid channels, rate, format, or frame count".into());
    }
    for sample in reader.samples::<f32>() {
        if !sample.map_err(|e| e.to_string())?.is_finite() {
            return Err("ctc WAV contains non-finite samples".into());
        }
    }
    Ok(reader.duration())
}

fn validate_fragment(fragment: &CtcFragment) -> Result<(), String> {
    let measurements = &fragment.measurements;
    let speakers: BTreeSet<_> = measurements.speakers.iter().collect();
    if !(1..=64).contains(&speakers.len())
        || speakers.len() != measurements.speakers.len()
        || measurements.files.len() != speakers.len()
        || measurements.head_positions.len() != 1
        || measurements.mics != CTC_MIC_ROLES
        || !(16_001..=384_000).contains(&fragment.sample_rate_hz)
        || !fragment.sweep_duration_s.is_finite()
        || fragment.sweep_duration_s <= 0.0
        || fragment.sweep_duration_s * f64::from(fragment.sample_rate_hz) > 16_000_000.0
        || !fragment.sweep_start_hz.is_finite()
        || !fragment.sweep_end_hz.is_finite()
        || fragment.sweep_start_hz <= 0.0
        || fragment.sweep_start_hz >= fragment.sweep_end_hz
        || fragment.sweep_end_hz >= f64::from(fragment.sample_rate_hz) / 2.0
    {
        return Err("invalid ctc matrix dimensions or sweep parameters".into());
    }
    let head = &measurements.head_positions[0];
    if head.id.trim().is_empty()
        || head.id.len() > 128
        || ![head.x, head.y, head.z, head.yaw_deg]
            .iter()
            .all(|value| value.is_finite())
        || speakers
            .iter()
            .any(|speaker| speaker.trim().is_empty() || speaker.len() > 128)
    {
        return Err("invalid ctc head or speaker identity".into());
    }
    let mut cells = BTreeSet::new();
    for entry in &measurements.files {
        if entry.head_position != head.id
            || !speakers.contains(&entry.speaker)
            || !cells.insert(&entry.speaker)
            || entry.ir.is_some()
        {
            return Err("ctc matrix contains unknown or duplicate cells".into());
        }
        for path in [entry.raw_sweep.as_deref(), entry.loopback.as_deref()] {
            crate::artifacts::member_name(path.ok_or("ctc cell is missing an audio member")?)?;
        }
    }
    crate::artifacts::member_name(&fragment.reference_sweep)?;
    Ok(())
}

/// Read a mono 32-bit float WAV fully.
fn read_mono_f32(path: &Path, sample_rate_hz: u32) -> Result<Vec<f32>, String> {
    let bytes = crate::artifacts::read_file(path, crate::artifacts::MAX_AUDIO_BYTES)?;
    validate_ctc_wave(&bytes, 1, sample_rate_hz)?;
    let mut reader =
        hound::WavReader::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?;
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
                let channels = if member == &entry.raw_sweep { 2 } else { 1 };
                crate::signal_recorder::write_wav_file(
                    &dir.join(member.as_deref().unwrap()),
                    &vec![0.1; 16 * usize::from(channels)],
                    48000,
                    channels,
                )
                .unwrap();
            }
        }
        crate::signal_recorder::write_wav_file(&dir.join(CTC_REFERENCE_FILE), &[0.1; 16], 48000, 1)
            .unwrap();
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
            sweep_duration_s: 16.0 / 48000.0,
            sweep_start_hz: 20.0,
            sweep_end_hz: 20000.0,
            sample_rate_hz: 48000,
            reference_sha256: crate::capture_session::calibration_sha256(
                &std::fs::read(dir.join(CTC_REFERENCE_FILE)).unwrap(),
            ),
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
    fn late_merge_failure_removes_staging_and_preserves_source_for_retry() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source");
        std::fs::create_dir(&source).unwrap();
        let fragment = fragment(&source, "head", &["left"]);
        write_fragment(&source, &fragment);
        let ears = fragment.measurements.files[0].raw_sweep.as_ref().unwrap();
        let original = std::fs::read(source.join(ears)).unwrap();
        let loopback = fragment.measurements.files[0].loopback.as_ref().unwrap();
        let saved_loopback = std::fs::read(source.join(loopback)).unwrap();
        // The valid ear file is copied first; the next member fails decoding.
        std::fs::write(source.join(loopback), b"truncated WAV").unwrap();
        let output = root.path().join("bundle");
        assert!(merge_ctc_fragments(std::slice::from_ref(&source), &output).is_err());
        assert!(!output.exists());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
        assert_eq!(std::fs::read(source.join(ears)).unwrap(), original);

        std::fs::write(source.join(loopback), saved_loopback).unwrap();
        let merged = merge_ctc_fragments(std::slice::from_ref(&source), &output).unwrap();
        for member in merged.files {
            assert!(output.join(member).is_file());
        }
        // A second publication cannot overwrite an already usable bundle.
        let descriptor = std::fs::read(output.join(CTC_MERGED_FILE)).unwrap();
        assert!(merge_ctc_fragments(std::slice::from_ref(&source), &output).is_err());
        assert_eq!(
            std::fs::read(output.join(CTC_MERGED_FILE)).unwrap(),
            descriptor
        );
    }

    #[test]
    fn merge_refuses_invalid_matrix_identities_and_reference_duration() {
        let root = tempfile::tempdir().unwrap();
        for mode in 0..4 {
            let mut fragment = fragment(root.path(), "head", &["left", "right"]);
            match mode {
                0 => fragment.measurements.files[1].speaker = "left".into(),
                1 => fragment.measurements.files[0].head_position = "other-head".into(),
                2 => fragment.sweep_duration_s = 1000.0,
                _ => fragment.sweep_start_hz = -1.0,
            }
            write_fragment(root.path(), &fragment);
            let output = root.path().join("merged");
            assert!(merge_ctc_fragments(&[root.path().to_path_buf()], &output).is_err());
            assert!(!output.exists());
        }
    }

    #[test]
    fn merge_rejects_escape_paths_for_every_member_without_publishing() {
        let root = tempfile::tempdir().unwrap();
        let outside = root.path().join("sentinel.wav");
        std::fs::write(&outside, b"private sentinel").unwrap();
        let source = root.path().join("session");
        std::fs::create_dir(&source).unwrap();
        for field in 0..3 {
            for name in [PathBuf::from("../sentinel.wav"), outside.clone()] {
                let mut fragment = fragment(&source, "head", &["left"]);
                match field {
                    0 => fragment.measurements.files[0].raw_sweep = Some(name),
                    1 => fragment.measurements.files[0].loopback = Some(name),
                    _ => fragment.reference_sweep = name,
                }
                write_fragment(&source, &fragment);
                let output = root.path().join("bundle");
                assert!(merge_ctc_fragments(std::slice::from_ref(&source), &output).is_err());
                assert!(!output.exists());
                assert_eq!(std::fs::read(&outside).unwrap(), b"private sentinel");
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn merge_refuses_symlinked_members_and_descriptor() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("session");
        std::fs::create_dir(&source).unwrap();
        for field in 0..4 {
            let fragment = fragment(&source, "head", &["left"]);
            write_fragment(&source, &fragment);
            let member = match field {
                0 => fragment.measurements.files[0].raw_sweep.as_deref().unwrap(),
                1 => fragment.measurements.files[0].loopback.as_deref().unwrap(),
                2 => &fragment.reference_sweep,
                _ => Path::new(CTC_FRAGMENT_FILE),
            };
            let target = root.path().join("outside");
            std::fs::copy(source.join(member), &target).unwrap();
            std::fs::remove_file(source.join(member)).unwrap();
            std::os::unix::fs::symlink(&target, source.join(member)).unwrap();
            let output = root.path().join("bundle");
            assert!(merge_ctc_fragments(std::slice::from_ref(&source), &output).is_err());
            assert!(!output.exists());
            std::fs::remove_file(source.join(member)).unwrap();
        }
    }

    #[test]
    fn merge_refuses_invalid_nonfinite_and_oversized_audio() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("session");
        std::fs::create_dir(&source).unwrap();
        for mode in 0..3 {
            let fragment = fragment(&source, "head", &["left"]);
            let member = source.join(fragment.measurements.files[0].raw_sweep.as_ref().unwrap());
            match mode {
                0 => std::fs::write(&member, b"not audio").unwrap(),
                1 => crate::signal_recorder::write_wav_file(&member, &[f32::NAN; 16], 48000, 2)
                    .unwrap(),
                _ => std::fs::OpenOptions::new()
                    .write(true)
                    .open(&member)
                    .unwrap()
                    .set_len(crate::artifacts::MAX_AUDIO_BYTES as u64 + 1)
                    .unwrap(),
            }
            write_fragment(&source, &fragment);
            let output = root.path().join("bundle");
            assert!(merge_ctc_fragments(std::slice::from_ref(&source), &output).is_err());
            assert!(!output.exists());
        }
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
