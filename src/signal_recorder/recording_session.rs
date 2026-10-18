use super::types::ChannelRecordingInfo;
use super::types::DeviceInfo;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Lightweight recording metadata (V2 format)
///
/// This format stores only metadata and file paths, with actual analysis
/// data stored in CSV files. This reduces JSON file size dramatically
/// (from ~90MB to ~2KB for a typical multi-channel recording session).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingSession {
    /// Format version (currently "2.0")
    pub version: String,
    /// Recording timestamp (RFC 3339 format)
    pub timestamp: String,
    /// Sample rate used for recording
    pub sample_rate: u32,
    /// Signal type used (sweep, pink-noise, etc.)
    pub signal_type: String,
    /// Signal duration in seconds
    pub signal_duration_secs: f32,
    /// Signal level in dBFS
    pub signal_level_db: f32,
    /// Sweep frequency range (if applicable)
    pub sweep_range: Option<(f32, f32)>,
    /// Playback device configuration
    pub playback_device: Option<DeviceInfo>,
    /// Recording device configuration
    pub recording_device: Option<DeviceInfo>,
    /// Microphone calibration file path (relative to session directory)
    pub mic_calibration_path: Option<String>,
    /// Per-channel microphone calibration file paths (parallel to channels)
    #[serde(default)]
    pub mic_calibration_paths: Vec<Option<String>>,
    /// Individual channel recordings
    pub channels: Vec<ChannelRecordingInfo>,
}

impl RecordingSession {
    /// Create a new recording session
    #[must_use]
    pub fn new(
        sample_rate: u32,
        signal_type: &str,
        signal_duration_secs: f32,
        signal_level_db: f32,
        sweep_range: Option<(f32, f32)>,
    ) -> Self {
        Self {
            version: "2.0".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            sample_rate,
            signal_type: signal_type.to_string(),
            signal_duration_secs,
            signal_level_db,
            sweep_range,
            playback_device: None,
            recording_device: None,
            mic_calibration_path: None,
            mic_calibration_paths: Vec::new(),
            channels: Vec::new(),
        }
    }

    /// Get the effective calibration path for a channel, checking per-channel first, then global fallback
    #[must_use]
    pub fn effective_calibration_for_channel(&self, idx: usize) -> Option<&str> {
        // Per-channel calibration takes priority
        if let Some(Some(path)) = self.mic_calibration_paths.get(idx)
            && !path.is_empty()
        {
            return Some(path.as_str());
        }
        // Fall back to global
        self.mic_calibration_path.as_deref()
    }

    /// Add a channel recording to the session
    #[allow(clippy::too_many_arguments)] // recording metadata has many independent fields
    pub fn add_channel(
        &mut self,
        channel_index: usize,
        channel_name: &str,
        output_channel: usize,
        input_channel: usize,
        wav_path: &str,
        csv_path: &str,
        success: bool,
        error: Option<String>,
    ) {
        self.channels.push(ChannelRecordingInfo {
            channel_index,
            channel_name: channel_name.to_string(),
            output_channel,
            input_channel,
            wav_path: wav_path.to_string(),
            csv_path: csv_path.to_string(),
            success,
            error,
            mic_calibration_path: None,
        });
    }

    /// Save session to JSON file
    ///
    /// # Errors
    /// Returns an error if the session cannot be serialized or the destination cannot be written.
    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path)
            .map_err(|e| format!("Failed to create session file: {e}"))?;
        serde_json::to_writer_pretty(file, self)
            .map_err(|e| format!("Failed to serialize session: {e}"))?;
        log::info!("[RecordingSession] Saved session to {}", path.display());
        Ok(())
    }

    /// Load session from JSON file
    ///
    /// # Errors
    /// Returns an error if the session file cannot be read or its JSON cannot be decoded.
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let bytes = crate::artifacts::read_file(path, crate::artifacts::MAX_METADATA_BYTES)?;
        let session: Self = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Failed to deserialize session: {e}"))?;
        validate_paths(path.parent().unwrap_or(Path::new(".")), &session)?;
        log::info!(
            "[RecordingSession] Loaded session from {} (version {})",
            path.display(),
            session.version
        );
        Ok(session)
    }
}

fn validate_paths(root: &Path, session: &RecordingSession) -> Result<(), String> {
    use crate::artifacts::member_path;
    if session.channels.len() > 64 {
        return Err("recording session exceeds 64 channels".into());
    }
    let mut inputs = std::collections::HashSet::new();
    let mut outputs = std::collections::HashSet::new();
    for channel in &session.channels {
        if !channel.success {
            continue;
        }
        member_path(root, Path::new(&channel.wav_path))?;
        member_path(root, Path::new(&channel.csv_path))?;
        if Path::new(&channel.csv_path)
            .extension()
            .and_then(|s| s.to_str())
            != Some("csv")
            || !outputs.insert(channel.csv_path.as_str())
        {
            return Err("recording output must be a unique local .csv file".into());
        }
        inputs.insert(channel.wav_path.as_str());
    }
    for name in session
        .mic_calibration_path
        .iter()
        .chain(session.mic_calibration_paths.iter().flatten())
        .chain(
            session
                .channels
                .iter()
                .filter_map(|channel| channel.mic_calibration_path.as_ref()),
        )
        .filter(|name| !name.is_empty())
    {
        member_path(root, Path::new(name))?;
        inputs.insert(name.as_str());
    }
    if outputs.iter().any(|name| inputs.contains(name)) {
        return Err("recording output would overwrite an input artifact".into());
    }
    Ok(())
}

fn snapshot(path: &Path, limit: usize) -> Result<tempfile::NamedTempFile, String> {
    use std::io::Write;
    let bytes = crate::artifacts::read_file(path, limit)?;
    let suffix = format!(
        ".{}",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("txt")
    );
    let mut copy = tempfile::NamedTempFile::with_suffix(&suffix).map_err(|e| e.to_string())?;
    copy.write_all(&bytes).map_err(|e| e.to_string())?;
    Ok(copy)
}

/// Re-process recordings from WAV files and regenerate CSV analysis files
///
/// This function loads WAV files from a recording session, re-runs the analysis,
/// and writes updated CSV files. Useful when analysis algorithms are updated.
///
/// # Arguments
/// * `session_dir` - Directory containing the recording session
/// * `session` - Recording session metadata
/// * `reference_signal` - Reference signal used for recording (must regenerate)
/// * `sample_rate` - Sample rate
/// * `sweep_range` - Sweep frequency range (if applicable)
/// * `mic_compensation_path` - Path to microphone calibration file (optional)
///
/// # Returns
/// Updated `RecordingSession` with new CSV paths
///
/// # Errors
/// Returns an error if the reference WAV cannot be prepared. Per-channel failures are recorded in the returned session.
#[expect(
    clippy::too_many_lines,
    reason = "Keep the ordered acquisition and artifact-processing phases together so cleanup and provenance remain explicit."
)]
pub fn reprocess_recordings(
    session_dir: &Path,
    session: &RecordingSession,
    reference_signal: &[f32],
    mic_compensation_path: Option<&Path>,
) -> Result<RecordingSession, String> {
    use math_audio_dsp::analysis::{MicrophoneCompensation, analyze_recording, write_analysis_csv};

    validate_paths(session_dir, session)?;
    if reference_signal.is_empty()
        || reference_signal.len() > 16_000_000
        || reference_signal.iter().any(|sample| !sample.is_finite())
    {
        return Err("invalid or oversized reference signal".into());
    }
    let load_compensation = |path: &Path| {
        let copy = snapshot(path, crate::artifacts::MAX_METADATA_BYTES)?;
        MicrophoneCompensation::from_file(copy.path())
    };
    log::info!(
        "[reprocess_recordings] Re-processing {} channels in {}",
        session.channels.len(),
        session_dir.display()
    );

    // Load global microphone compensation (used as fallback)
    let global_compensation = if let Some(comp_path) = mic_compensation_path {
        Some(load_compensation(comp_path)?)
    } else if let Some(ref rel_path) = session.mic_calibration_path {
        let full_path = session_dir.join(rel_path);
        if full_path.exists() {
            Some(load_compensation(&full_path)?)
        } else {
            None
        }
    } else {
        None
    };

    let mut updated_session = session.clone();
    updated_session.channels.clear();

    for (ch_idx, channel_info) in session.channels.iter().enumerate() {
        if !channel_info.success {
            // Keep failed channels as-is
            updated_session.channels.push(channel_info.clone());
            continue;
        }

        let wav_path = session_dir.join(&channel_info.wav_path);
        let csv_path = session_dir.join(&channel_info.csv_path);

        if !wav_path.exists() {
            log::warn!(
                "[reprocess_recordings] WAV file not found: {}, skipping channel {}",
                wav_path.display(),
                channel_info.channel_name
            );
            let mut failed_channel = channel_info.clone();
            failed_channel.success = false;
            failed_channel.error = Some(format!("WAV file not found: {}", wav_path.display()));
            updated_session.channels.push(failed_channel);
            continue;
        }

        log::info!(
            "[reprocess_recordings] Processing channel '{}' from {}",
            channel_info.channel_name,
            wav_path.display()
        );

        // Resolve per-channel compensation with fallback chain:
        // 1. ChannelRecordingInfo.mic_calibration_path (per-recording override)
        // 2. RecordingSession.mic_calibration_paths[idx] (per-channel session config)
        // 3. Global compensation (from mic_compensation_path arg or session.mic_calibration_path)
        let per_channel_cal_path = channel_info
            .mic_calibration_path
            .as_deref()
            .or_else(|| {
                session
                    .mic_calibration_paths
                    .get(ch_idx)
                    .and_then(|p| p.as_deref())
            })
            .filter(|p| !p.is_empty());

        let channel_compensation = if let Some(cal_path) = per_channel_cal_path {
            let ch_path = Path::new(cal_path);
            let full_path = if ch_path.is_absolute() {
                ch_path.to_path_buf()
            } else {
                session_dir.join(ch_path)
            };
            if full_path.exists() {
                Some(load_compensation(&full_path)?)
            } else {
                log::warn!(
                    "[reprocess_recordings] Per-channel calibration file not found: {}, using global",
                    full_path.display()
                );
                global_compensation.clone()
            }
        } else {
            global_compensation.clone()
        };

        // Re-analyze the recording
        // Analyze a bounded private snapshot, so a concurrent link replacement
        // cannot redirect the analysis library's later path-based read.
        let wav_snapshot = snapshot(&wav_path, crate::artifacts::MAX_AUDIO_BYTES)?;
        match analyze_recording(
            wav_snapshot.path(),
            reference_signal,
            session.sample_rate,
            session.sweep_range,
        ) {
            Ok(analysis) => {
                // Rename a private temporary file over the destination; never
                // follow a destination symlink substituted after preflight.
                let write_csv = || -> Result<(), String> {
                    let temporary =
                        tempfile::NamedTempFile::new_in(session_dir).map_err(|e| e.to_string())?;
                    write_analysis_csv(&analysis, temporary.path(), channel_compensation.as_ref())?;
                    temporary.persist(&csv_path).map_err(|e| e.to_string())?;
                    Ok(())
                };
                if let Err(e) = write_csv() {
                    log::error!(
                        "[reprocess_recordings] Failed to write CSV for channel '{}': {}",
                        channel_info.channel_name,
                        e
                    );
                    let mut failed_channel = channel_info.clone();
                    failed_channel.success = false;
                    failed_channel.error = Some(format!("Failed to write CSV: {e}"));
                    updated_session.channels.push(failed_channel);
                } else {
                    log::info!(
                        "[reprocess_recordings] Updated CSV for channel '{}'",
                        channel_info.channel_name
                    );
                    updated_session.channels.push(channel_info.clone());
                }
            }
            Err(e) => {
                log::error!(
                    "[reprocess_recordings] Analysis failed for channel '{}': {}",
                    channel_info.channel_name,
                    e
                );
                let mut failed_channel = channel_info.clone();
                failed_channel.success = false;
                failed_channel.error = Some(format!("Analysis failed: {e}"));
                updated_session.channels.push(failed_channel);
            }
        }
    }

    Ok(updated_session)
}

#[cfg(test)]
mod audit_tests {
    use super::*;

    fn session() -> RecordingSession {
        let mut session = RecordingSession::new(48000, "sweep", 0.1, -20.0, Some((20.0, 20000.0)));
        session.add_channel(0, "left", 0, 0, "take.wav", "take.csv", true, None);
        session
    }

    #[test]
    fn local_recording_and_text_calibration_reprocess_successfully() {
        let root = tempfile::tempdir().unwrap();
        let reference =
            math_audio_dsp::signals::try_gen_log_sweep(20.0, 20000.0, 0.1, 48000, 0.1).unwrap();
        crate::signal_recorder::write_wav_file(&root.path().join("take.wav"), &reference, 48000, 1)
            .unwrap();
        std::fs::write(root.path().join("cal.txt"), "20 0\n1000 0\n20000 0\n").unwrap();
        let mut session = session();
        session.mic_calibration_path = Some("cal.txt".into());
        let result = reprocess_recordings(root.path(), &session, &reference, None).unwrap();
        assert!(result.channels[0].success, "{:?}", result.channels[0].error);
        assert!(root.path().join("take.csv").is_file());
        let snapshot = snapshot(&root.path().join("cal.txt"), 1024).unwrap();
        let calibration =
            math_audio_dsp::analysis::MicrophoneCompensation::from_file(snapshot.path()).unwrap();
        assert_eq!(calibration.frequencies.first().copied(), Some(20.0));
    }

    #[test]
    fn refuses_traversal_and_absolute_outputs_before_overwriting_sentinel() {
        let root = tempfile::tempdir().unwrap();
        let capture = root.path().join("capture");
        std::fs::create_dir(&capture).unwrap();
        let outside = root.path().join("sentinel.csv");
        std::fs::write(&outside, b"keep me").unwrap();
        for name in [
            "../sentinel.csv".to_string(),
            outside.to_string_lossy().into_owned(),
        ] {
            let mut session = session();
            session.channels[0].csv_path = name;
            assert!(reprocess_recordings(&capture, &session, &[0.1; 16], None).is_err());
            let metadata = capture.join("recordings.json");
            session.save_to_file(&metadata).unwrap();
            assert!(RecordingSession::load_from_file(&metadata).is_err());
            assert_eq!(std::fs::read(&outside).unwrap(), b"keep me");
        }
    }

    #[test]
    fn refuses_input_escape_duplicate_outputs_and_calibration_overwrite() {
        let root = tempfile::tempdir().unwrap();
        let mut session = session();
        session.channels[0].wav_path = "../outside.wav".into();
        assert!(validate_paths(root.path(), &session).is_err());
        session.channels[0].wav_path = "take.wav".into();
        session.mic_calibration_path = Some("../outside.txt".into());
        assert!(validate_paths(root.path(), &session).is_err());
        session.mic_calibration_path = Some("take.csv".into());
        assert!(validate_paths(root.path(), &session).is_err());
        session.mic_calibration_path = None;
        session.channels.push(session.channels[0].clone());
        assert!(validate_paths(root.path(), &session).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn refuses_symlinked_input_output_and_calibration() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        for name in ["take.wav", "take.csv", "cal.txt"] {
            let mut session = session();
            session.mic_calibration_path = Some("cal.txt".into());
            std::os::unix::fs::symlink(outside.path(), root.path().join(name)).unwrap();
            assert!(reprocess_recordings(root.path(), &session, &[0.1; 16], None).is_err());
            std::fs::remove_file(root.path().join(name)).unwrap();
        }
    }
}
