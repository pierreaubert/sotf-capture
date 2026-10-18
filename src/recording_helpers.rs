//! Shared recording-session helpers for all player frontends (TUI, GPUI, CLI).
//!
//! Consolidates the per-frontend recording logic flagged in
//! `reviews/20260818-recording.md` (B1, B3, B4, B5, C1) so every shell maps
//! signal types, converts levels, sanitizes file names, resolves mic
//! calibration, and builds capture parameters identically. The
//! `RecordingConfiguration` builder lives on
//! [`crate::wizard::RecordingScreenModel::build_recording_configuration`]
//! next to the other save-time helpers.

// Rust guideline compliant 2026-02-21

use crate::quality::{
    QualityFinding, QualityMethod, QualityMetric, QualityStatus, QualityUnit, ThresholdComparison,
};
use crate::recording_types::{
    ChannelRecording, ChannelRecordingState, RecordingResult, RecordingSignalType,
    RecordingSourceIdentity, TakeQualitySummary,
};
use crate::signal_recorder::{
    CaptureAnalysis, DEFAULT_MLS_ORDER, MIN_REPEAT_SWEEPS, SignalParams, SignalType,
    sweep_params_from_config,
};
use std::fmt::Write as _;

/// Canonical filename for the saved recording session JSON (B5).
///
/// Room EQ's "`FromFile`" flow and the i18n strings point users at this name,
/// so every frontend must save the session as `<dir>/recordings.json`
/// regardless of the user-chosen session name.
pub const RECORDINGS_FILENAME: &str = "recordings.json";

/// Adjacent, versioned sidecar for quality evidence associated with
/// [`RECORDINGS_FILENAME`]. The canonical `RoomConfig` remains unchanged.
pub const RECORDING_QUALITY_FILENAME: &str = "recording-quality.json";

/// Current version of the recording quality sidecar schema.
pub const RECORDING_QUALITY_SIDECAR_VERSION: u32 = 1;

/// One saved source and the measured quality evidence for that take.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RecordingQualityEntry {
    /// Stable `RoomConfig` identity, independent of display ordering.
    pub source: RecordingSourceIdentity,
    pub quality: TakeQualitySummary,
}

/// Effective stimulus settings and playback length for one capture source.
///
/// `generated_frames` includes stimulus-internal silence windows, while
/// `playback_frames` also includes the standard playback padding. This keeps
/// the configured duration distinct from the duration the recorder plays.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EffectiveAcquisitionConfig {
    /// Signal kind used by the capture generator.
    pub signal_type: String,
    /// Playback sample rate in hertz.
    pub sample_rate_hz: u32,
    /// Nominal duration supplied to the generator. Some signals ignore it.
    pub requested_signal_duration_secs: f32,
    /// Whether the selected generator uses the nominal duration.
    pub duration_control_applies: bool,
    /// Number of samples returned by the signal generator before outer padding.
    pub generated_frames: usize,
    /// Duration represented by `generated_frames`.
    pub generated_duration_secs: f32,
    /// Number of outer silence-padding samples added before and after playback.
    pub playback_padding_frames: usize,
    /// Total samples played by the capture path.
    pub playback_frames: usize,
    /// Duration represented by `playback_frames`.
    pub playback_duration_secs: f32,
    /// Peak linear amplitude passed to the signal generator.
    pub amplitude_peak: Option<f32>,
    /// Equivalent peak dBFS when the amplitude is positive and finite.
    pub level_dbfs: Option<f32>,
    /// Lowest stimulus frequency when the signal has a frequency bound.
    pub low_frequency_hz: Option<f32>,
    /// Highest stimulus frequency when the signal has a frequency bound.
    pub high_frequency_hz: Option<f32>,
    /// Seconds per octave below 100 Hz for octave-scaled sweeps.
    pub bass_octave_duration_secs: Option<f32>,
    /// Internal pre-sweep silence duration for octave-scaled sweeps.
    pub pre_silence_secs: Option<f32>,
    /// Internal post-sweep silence duration for octave-scaled sweeps.
    pub post_silence_secs: Option<f32>,
}

/// Describe the exact frame duration and signal settings used by capture.
///
/// The calculation mirrors the shared generator: fixed-duration signals use
/// the requested duration, MLS uses its sequence length, and octave-scaled
/// sweeps use their bass-weighted duration plus their configured silence.
///
/// # Examples
///
/// ```
/// use sotf_capture::recording_helpers::effective_acquisition_config;
/// use sotf_capture::signal_recorder::{SignalParams, SignalType};
///
/// let params = SignalParams::Sweep {
///     start_freq: 20.0,
///     end_freq: 20_000.0,
///     amp: 0.5,
/// };
/// let config = effective_acquisition_config(SignalType::Sweep, &params, 2.0, 48_000)?;
/// assert_eq!(config.generated_frames, 96_000);
/// assert_eq!(config.playback_frames, 120_000);
/// # Ok::<(), String>(())
/// ```
///
/// # Errors
/// Returns an error for mismatched signal parameters, invalid duration, or a
/// zero sample rate.
///
/// # Panics
/// This function does not panic for caller-provided signal settings.
#[expect(
    clippy::too_many_lines,
    reason = "Keep the ordered acquisition and artifact-processing phases together so cleanup and provenance remain explicit."
)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "Validated durations round to nonnegative frame counts; displayed durations follow the f32 signal-generator API."
)]
pub fn effective_acquisition_config(
    signal_type: SignalType,
    params: &SignalParams,
    requested_signal_duration_secs: f32,
    sample_rate_hz: u32,
) -> Result<EffectiveAcquisitionConfig, String> {
    if sample_rate_hz == 0 {
        return Err("sample rate must be greater than zero".to_string());
    }
    if !requested_signal_duration_secs.is_finite()
        || (signal_type != SignalType::Mls && requested_signal_duration_secs <= 0.0)
    {
        return Err("signal duration must be finite and positive when used".to_string());
    }
    let finite_parameters = match params {
        SignalParams::Tone { freq, amp } => freq.is_finite() && amp.is_finite(),
        SignalParams::TwoTone {
            freq1,
            amp1,
            freq2,
            amp2,
        } => [freq1, amp1, freq2, amp2]
            .into_iter()
            .all(|value| value.is_finite()),
        SignalParams::Sweep {
            start_freq,
            end_freq,
            amp,
        } => [start_freq, end_freq, amp]
            .into_iter()
            .all(|value| value.is_finite()),
        SignalParams::OctaveSweep {
            start_freq,
            end_freq,
            amp,
            bass_octave_duration_s,
            pre_silence_s,
            post_silence_s,
        } => [
            start_freq,
            end_freq,
            amp,
            bass_octave_duration_s,
            pre_silence_s,
            post_silence_s,
        ]
        .into_iter()
        .all(|value| value.is_finite()),
        SignalParams::Noise { amp }
        | SignalParams::Mls { amp, .. }
        | SignalParams::Dirac { amp } => amp.is_finite(),
        SignalParams::Stipa => true,
    };
    if !finite_parameters {
        return Err("signal parameters must be finite".to_string());
    }
    crate::signal_recorder::validate_signal_params(
        signal_type,
        params,
        requested_signal_duration_secs,
        sample_rate_hz,
    )?;

    let mut config = EffectiveAcquisitionConfig {
        signal_type: signal_type.as_str().to_string(),
        sample_rate_hz,
        requested_signal_duration_secs,
        duration_control_applies: true,
        generated_frames: 0,
        generated_duration_secs: 0.0,
        playback_padding_frames: 2 * ((0.250_f32 * sample_rate_hz as f32) as usize),
        playback_frames: 0,
        playback_duration_secs: 0.0,
        amplitude_peak: None,
        level_dbfs: None,
        low_frequency_hz: None,
        high_frequency_hz: None,
        bass_octave_duration_secs: None,
        pre_silence_secs: None,
        post_silence_secs: None,
    };

    let sample_rate = f64::from(sample_rate_hz);
    config.generated_frames = match (signal_type, params) {
        (
            SignalType::Sweep,
            SignalParams::Sweep {
                start_freq,
                end_freq,
                amp,
            },
        ) => {
            config.amplitude_peak = Some(*amp);
            config.low_frequency_hz = Some(*start_freq);
            config.high_frequency_hz = Some(*end_freq);
            math_audio_dsp::signals::frames_for(requested_signal_duration_secs, sample_rate_hz)
        }
        (
            SignalType::Sweep,
            SignalParams::OctaveSweep {
                start_freq,
                end_freq,
                amp,
                bass_octave_duration_s,
                pre_silence_s,
                post_silence_s,
            },
        ) => {
            config.duration_control_applies = false;
            config.amplitude_peak = Some(*amp);
            config.low_frequency_hz = Some(*start_freq);
            config.high_frequency_hz = Some(*end_freq);
            let bass_duration = bass_octave_duration_s.clamp(1.0, 10.0);
            let pre_silence = pre_silence_s.max(0.0);
            let post_silence = post_silence_s.max(0.0);
            config.bass_octave_duration_secs = Some(bass_duration);
            config.pre_silence_secs = Some(pre_silence);
            config.post_silence_secs = Some(post_silence);

            // Keep these boundaries and rate ratios aligned with
            // `gen_log_sweep_octave_scaled` in math-audio.
            let low = f64::from(*start_freq).max(1.0);
            let high = f64::from(*end_freq).max(low * 1.001);
            let bass_rate = f64::from(bass_duration);
            let bass_octaves = octave_span(low, 100.0_f64.min(high));
            let mid_octaves = octave_span(100.0_f64.max(low), 1_000.0_f64.min(high));
            let high_octaves = octave_span(1_000.0_f64.max(low), high);
            let sweep_duration = (bass_octaves * bass_rate
                + mid_octaves * bass_rate * 0.5
                + high_octaves * bass_rate * 0.25)
                .max(1.0);
            let sweep_frames = (sweep_duration * sample_rate).round() as usize;
            let pre_frames = (pre_silence * sample_rate_hz as f32).round() as usize;
            let post_frames = (post_silence * sample_rate_hz as f32).round() as usize;
            pre_frames
                .saturating_add(sweep_frames)
                .saturating_add(post_frames)
        }
        (SignalType::Tone, SignalParams::Tone { freq, amp }) => {
            config.amplitude_peak = Some(*amp);
            config.low_frequency_hz = Some(*freq);
            config.high_frequency_hz = Some(*freq);
            math_audio_dsp::signals::frames_for(requested_signal_duration_secs, sample_rate_hz)
        }
        (
            SignalType::TwoTone,
            SignalParams::TwoTone {
                freq1,
                freq2,
                amp1,
                amp2,
            },
        ) => {
            config.amplitude_peak = Some(amp1.abs().max(amp2.abs()));
            config.low_frequency_hz = Some((*freq1).min(*freq2));
            config.high_frequency_hz = Some((*freq1).max(*freq2));
            math_audio_dsp::signals::frames_for(requested_signal_duration_secs, sample_rate_hz)
        }
        (
            SignalType::WhiteNoise | SignalType::PinkNoise | SignalType::MNoise,
            SignalParams::Noise { amp },
        )
        | (SignalType::Dirac, SignalParams::Dirac { amp }) => {
            config.amplitude_peak = Some(*amp);
            math_audio_dsp::signals::frames_for(requested_signal_duration_secs, sample_rate_hz)
        }
        (SignalType::Mls, SignalParams::Mls { order, amp }) => {
            config.duration_control_applies = false;
            config.amplitude_peak = Some(*amp);
            (1_usize << u32::from(*order)) - 1
        }
        (SignalType::Stipa, SignalParams::Stipa) => {
            math_audio_dsp::signals::frames_for(requested_signal_duration_secs, sample_rate_hz)
        }
        _ => {
            return Err(format!(
                "signal type {signal_type:?} does not match parameters {params:?}"
            ));
        }
    };
    config.generated_duration_secs = config.generated_frames as f32 / sample_rate_hz as f32;
    config.playback_frames = config
        .generated_frames
        .saturating_add(config.playback_padding_frames);
    config.playback_duration_secs = config.playback_frames as f32 / sample_rate_hz as f32;
    if let Some(amplitude_peak) = config.amplitude_peak
        && amplitude_peak.is_finite()
        && amplitude_peak > 0.0
    {
        config.level_dbfs = Some(20.0 * amplitude_peak.log10());
    }
    Ok(config)
}

fn octave_span(low_hz: f64, high_hz: f64) -> f64 {
    if high_hz > low_hz {
        (high_hz / low_hz).log2()
    } else {
        0.0
    }
}

/// Per-source stimulus provenance stored alongside quality evidence.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RecordingStimulusEntry {
    /// Stable `RoomConfig` source identity.
    pub source: RecordingSourceIdentity,
    /// Effective signal configuration used for this source.
    pub config: EffectiveAcquisitionConfig,
}

/// Quality evidence persisted next to `recordings.json` without extending
/// `InlineMeasurement` or `RoomConfig` serialization.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RecordingQualitySidecar {
    pub version: u32,
    /// SHA-256 of the exact `recordings.json` bytes. Stale sidecars are
    /// ignored if the session was replaced without a matching evidence write.
    pub session_sha256: String,
    /// Filesystem write timestamp of the exact canonical file version. This
    /// also detects a same-content session rewrite with different evidence.
    pub session_modified_unix_nanos: u64,
    pub entries: Vec<RecordingQualityEntry>,
    /// Optional per-source signal settings; absent in older sidecars.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stimuli: Vec<RecordingStimulusEntry>,
}

impl RecordingQualitySidecar {
    /// Find evidence for a source, if the sidecar contains that exact source.
    #[must_use]
    pub fn quality_for(&self, source: &RecordingSourceIdentity) -> Option<&TakeQualitySummary> {
        self.entries
            .iter()
            .find(|entry| &entry.source == source)
            .map(|entry| &entry.quality)
    }

    /// Find the signal settings recorded for one source.
    #[must_use]
    pub fn stimulus_for(
        &self,
        source: &RecordingSourceIdentity,
    ) -> Option<&EffectiveAcquisitionConfig> {
        self.stimuli
            .iter()
            .find(|entry| &entry.source == source)
            .map(|entry| &entry.config)
    }

    /// Replace or append one source's settings while retaining its neighbors.
    pub fn upsert_stimulus(&mut self, entry: RecordingStimulusEntry) {
        if let Some(existing) = self
            .stimuli
            .iter_mut()
            .find(|existing| existing.source == entry.source)
        {
            *existing = entry;
        } else {
            self.stimuli.push(entry);
        }
    }

    /// Replace or append one source's quality while retaining its neighbors.
    pub fn upsert_quality(&mut self, entry: RecordingQualityEntry) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|existing| existing.source == entry.source)
        {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
    }
}

/// Atomically save versioned quality evidence beside a canonical session.
/// Include every source that has evidence; an empty entry list deliberately
/// replaces an old sidecar rather than allowing stale results to survive.
///
/// # Errors
/// Returns an error if the destination cannot be created or the metadata cannot be serialized or written.
pub fn save_recording_quality_sidecar(
    recordings_json_path: &std::path::Path,
    recordings_json: &[u8],
    entries: Vec<RecordingQualityEntry>,
) -> std::io::Result<()> {
    let stimuli = load_recording_quality_sidecar(recordings_json_path, recordings_json)?
        .map_or_else(Vec::new, |sidecar| sidecar.stimuli);
    save_recording_quality_sidecar_with_stimuli(
        recordings_json_path,
        recordings_json,
        entries,
        stimuli,
    )
}

/// Atomically save quality findings and per-source stimulus provenance.
///
/// The canonical `recordings.json` remains unchanged. Duplicate identities
/// are rejected independently in the quality and stimulus collections.
///
/// # Errors
/// Returns an error if the destination cannot be created or the metadata cannot be serialized or written.
pub fn save_recording_quality_sidecar_with_stimuli(
    recordings_json_path: &std::path::Path,
    recordings_json: &[u8],
    entries: Vec<RecordingQualityEntry>,
    stimuli: Vec<RecordingStimulusEntry>,
) -> std::io::Result<()> {
    let current_session = std::fs::read(recordings_json_path)?;
    if current_session != recordings_json {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "recordings.json changed before its quality sidecar was written",
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(entries.len());
    if entries
        .iter()
        .any(|entry| !seen.insert(entry.source.clone()))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "quality sidecar contains duplicate source identities",
        ));
    }
    let mut seen_stimuli = std::collections::HashSet::with_capacity(stimuli.len());
    if stimuli
        .iter()
        .any(|entry| !seen_stimuli.insert(entry.source.clone()))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "quality sidecar contains duplicate stimulus identities",
        ));
    }
    let sidecar = RecordingQualitySidecar {
        version: RECORDING_QUALITY_SIDECAR_VERSION,
        session_sha256: recording_session_sha256(recordings_json),
        session_modified_unix_nanos: recording_session_modified_unix_nanos(recordings_json_path)?,
        entries,
        stimuli,
    };
    let json = serde_json::to_vec_pretty(&sidecar)
        .map_err(|error| std::io::Error::other(format!("serialize quality sidecar: {error}")))?;
    let path = recording_quality_sidecar_path(recordings_json_path);
    save_recording_session_json(&path, &json)
}

/// Load the adjacent sidecar for these exact session bytes.
///
/// Missing, unsupported-version, or mismatched sidecars return `Ok(None)`;
/// callers must leave quality as `None` in those cases rather than infer a
/// clean result. Malformed sidecar data returns `InvalidData` so the caller
/// can surface or log the damaged evidence file.
///
/// # Errors
/// Returns an error for unreadable or malformed metadata. Missing or stale sidecars return `Ok(None)`.
pub fn load_recording_quality_sidecar(
    recordings_json_path: &std::path::Path,
    recordings_json: &[u8],
) -> std::io::Result<Option<RecordingQualitySidecar>> {
    let current_session = match std::fs::read(recordings_json_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if current_session != recordings_json {
        return Ok(None);
    }
    let path = recording_quality_sidecar_path(recordings_json_path);
    let json = match std::fs::read(path) {
        Ok(json) => json,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let sidecar: RecordingQualitySidecar = serde_json::from_slice(&json).map_err(|error| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("parse recording quality sidecar: {error}"),
        )
    })?;
    if sidecar.version != RECORDING_QUALITY_SIDECAR_VERSION
        || sidecar.session_sha256 != recording_session_sha256(recordings_json)
        || recording_session_modified_unix_nanos(recordings_json_path).ok()
            != Some(sidecar.session_modified_unix_nanos)
    {
        return Ok(None);
    }
    let mut seen = std::collections::HashSet::with_capacity(sidecar.entries.len());
    if sidecar
        .entries
        .iter()
        .any(|entry| !seen.insert(entry.source.clone()))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "quality sidecar contains duplicate source identities",
        ));
    }
    let mut seen_stimuli = std::collections::HashSet::with_capacity(sidecar.stimuli.len());
    if sidecar
        .stimuli
        .iter()
        .any(|entry| !seen_stimuli.insert(entry.source.clone()))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "quality sidecar contains duplicate stimulus identities",
        ));
    }
    Ok(Some(sidecar))
}

fn recording_quality_sidecar_path(recordings_json_path: &std::path::Path) -> std::path::PathBuf {
    recordings_json_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(RECORDING_QUALITY_FILENAME)
}

fn recording_session_sha256(recordings_json: &[u8]) -> String {
    use sha2::Digest as _;
    use std::fmt::Write as _;

    let digest = sha2::Sha256::digest(recordings_json);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hex, "{byte:02x}").expect("writing to a String cannot fail");
    }
    hex
}

fn recording_session_modified_unix_nanos(path: &std::path::Path) -> std::io::Result<u64> {
    use std::time::UNIX_EPOCH;

    std::fs::metadata(path)?
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid recordings.json timestamp: {error}"),
            )
        })?
        .as_nanos()
        .try_into()
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "recordings.json timestamp exceeds sidecar format",
            )
        })
}

/// Build sidecar entries using the same identity ordering as
/// `build_speakers_from_recordings` in sotf-player. `speaker_names_by_channel`
/// must be the channel-name list supplied to that builder. Imported rows
/// retain their original source identity instead of being renumbered.
#[must_use]
pub fn recording_quality_entries_for_sources(
    recordings: &[ChannelRecording],
    speaker_names_by_channel: &[String],
) -> Vec<RecordingQualityEntry> {
    recording_source_recordings(recordings, speaker_names_by_channel)
        .into_iter()
        .filter_map(|(source, index)| {
            let recording = &recordings[index];
            let quality = recording.result.as_ref()?.quality.as_ref()?.clone();
            Some(RecordingQualityEntry { source, quality })
        })
        .collect()
}

/// Build per-source stimulus entries using the same source ordering as quality entries.
///
/// The callback can return a different acquisition snapshot for each source,
/// which preserves actual per-take gain and timing when only one source is
/// re-recorded after a setting change.
pub fn recording_stimulus_entries_for_sources(
    recordings: &[ChannelRecording],
    speaker_names_by_channel: &[String],
    mut config_for_source: impl FnMut(
        &RecordingSourceIdentity,
        &ChannelRecording,
    ) -> Option<EffectiveAcquisitionConfig>,
) -> Vec<RecordingStimulusEntry> {
    recording_source_recordings(recordings, speaker_names_by_channel)
        .into_iter()
        .filter_map(|(source, index)| {
            let config = config_for_source(&source, &recordings[index])?;
            Some(RecordingStimulusEntry { source, config })
        })
        .collect()
}

/// Build source-keyed stimulus entries from snapshots stored on each take.
#[must_use]
pub fn recording_stimulus_entries_from_sources(
    recordings: &[ChannelRecording],
    speaker_names_by_channel: &[String],
) -> Vec<RecordingStimulusEntry> {
    recording_stimulus_entries_for_sources(recordings, speaker_names_by_channel, |_, recording| {
        recording
            .result
            .as_ref()
            .and_then(|result| result.stimulus_config.clone())
    })
}

/// Restore missing metrics from analysis CSVs on the matching frequency grid.
///
/// The session JSON keeps only the frequency response, so a loaded
/// take arrives with `thd_percent`/`rt60_ms`/clarity/group-delay
/// unset and the Evaluating panels showed "not available" for a
/// measurement that was fully analyzed at capture time. The CSV
/// written next to the WAV at capture carries those exact series on
/// the same frequency grid; fill only the missing fields and only
/// when the grids line up.
pub fn enrich_results_from_take_csvs(recordings: &mut [ChannelRecording]) {
    for recording in recordings.iter_mut() {
        let Some(result) = recording.result.as_mut() else {
            continue;
        };
        if result.thd_percent.is_some()
            && result.rt60_ms.is_some()
            && result.clarity_c50_db.is_some()
            && result.clarity_c80_db.is_some()
            && result.excess_group_delay_ms.is_some()
        {
            continue;
        }
        let Some(csv_path) = result.csv_path.clone() else {
            continue;
        };
        // Parse a bounded private copy, including if the source grows or is
        // replaced while a frontend is loading the session.
        let Ok(bytes) = crate::artifacts::read_file(
            std::path::Path::new(&csv_path),
            crate::artifacts::MAX_METADATA_BYTES,
        ) else {
            continue;
        };
        let Ok(mut snapshot) = tempfile::NamedTempFile::with_suffix(".csv") else {
            continue;
        };
        if std::io::Write::write_all(&mut snapshot, &bytes).is_err() {
            continue;
        }
        let Ok(parsed) = math_audio_dsp::analysis::read_analysis_csv(snapshot.path()) else {
            continue;
        };
        if parsed.frequencies.len() != result.frequencies.len()
            || parsed
                .frequencies
                .iter()
                .zip(result.frequencies.iter())
                .any(|(a, b)| !a.is_finite() || !b.is_finite() || (a - b).abs() > 0.01)
        {
            continue;
        }
        for (destination, values) in [
            (&mut result.thd_percent, &parsed.thd_percent),
            (&mut result.rt60_ms, &parsed.rt60_ms),
            (&mut result.clarity_c50_db, &parsed.clarity_c50_db),
            (&mut result.clarity_c80_db, &parsed.clarity_c80_db),
            (
                &mut result.excess_group_delay_ms,
                &parsed.excess_group_delay_ms,
            ),
        ] {
            if destination.is_none()
                && values.len() == result.frequencies.len()
                && values.iter().any(|value| value.is_finite())
            {
                *destination = Some(
                    values
                        .iter()
                        .map(|value| if value.is_finite() { *value } else { f32::NAN })
                        .collect(),
                );
            }
        }
    }
}

/// Restore each source's stimulus snapshot from an exact-session sidecar.
///
/// Sources missing from the sidecar retain unknown acquisition settings.
pub fn restore_recording_stimuli_from_sidecar(
    recordings: &mut [ChannelRecording],
    speaker_names_by_channel: &[String],
    sidecar: &RecordingQualitySidecar,
) {
    let source_recordings = recording_source_recordings(recordings, speaker_names_by_channel);
    for (source, index) in source_recordings {
        if let Some(result) = recordings
            .get_mut(index)
            .and_then(|recording| recording.result.as_mut())
        {
            result.stimulus_config = sidecar.stimulus_for(&source).cloned();
        }
    }
}

fn recording_source_recordings(
    recordings: &[ChannelRecording],
    speaker_names_by_channel: &[String],
) -> Vec<(RecordingSourceIdentity, usize)> {
    use std::collections::BTreeMap;

    let mut sources = Vec::new();
    let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, recording) in recordings.iter().enumerate() {
        if recording.state != ChannelRecordingState::Done || recording.result.is_none() {
            continue;
        }
        if let Some(source) = recording.imported_source.as_ref() {
            sources.push((source.clone(), index));
        } else if recording
            .result
            .as_ref()
            .is_some_and(|result| result.wav_path.is_some() || result.csv_path.is_some())
        {
            grouped
                .entry(recording.channel_index)
                .or_default()
                .push(index);
        }
    }

    for (channel_index, mut group) in grouped {
        group.sort_by_key(|index| {
            let recording = &recordings[*index];
            (recording.mic_position_index, recording.mic_index)
        });
        let speaker = speaker_names_by_channel
            .get(channel_index)
            .map(String::as_str)
            .or_else(|| {
                group.first().map(|index| {
                    let recording = &recordings[*index];
                    recording
                        .channel_name
                        .find(" (")
                        .map_or(recording.channel_name.as_str(), |position| {
                            &recording.channel_name[..position]
                        })
                })
            });
        let Some(speaker) = speaker else {
            continue;
        };
        for (measurement_index, index) in group.into_iter().enumerate() {
            sources.push((
                RecordingSourceIdentity {
                    speaker: speaker.to_string(),
                    driver_index: None,
                    measurement_index,
                },
                index,
            ));
        }
    }
    sources
}

/// Replace session metadata only after a complete write succeeds.
/// A failed write leaves the previous saved session intact.
///
/// # Errors
/// Returns an error if the destination cannot be created or the metadata cannot be serialized or written.
pub fn save_recording_session_json(path: &std::path::Path, json: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    replace_recording_session(path, |file| file.write_all(json))
}

fn replace_recording_session(
    path: &std::path::Path,
    write: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    write(temporary.as_file_mut())?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(())
}

/// Shared default sweep start frequency (Hz) for all frontends (C2).
pub const DEFAULT_SWEEP_START_FREQ: f32 = 20.0;

/// Shared default sweep end frequency (Hz) for all frontends (C2).
pub const DEFAULT_SWEEP_END_FREQ: f32 = 20000.0;

/// Map the wizard's [`RecordingSignalType`] to the engine's [`SignalType`].
///
/// `DelayProbe` has no per-channel generator — it uses the separate
/// multi-channel `probe_channel_delays` workflow — so it falls back to
/// `Sweep` here, matching what both frontends did inline.
#[must_use]
pub fn signal_type_for(recording_signal_type: RecordingSignalType) -> SignalType {
    match recording_signal_type {
        RecordingSignalType::Sweep => SignalType::Sweep,
        RecordingSignalType::WhiteNoise => SignalType::WhiteNoise,
        RecordingSignalType::PinkNoise => SignalType::PinkNoise,
        RecordingSignalType::Mls => SignalType::Mls,
        RecordingSignalType::Dirac => SignalType::Dirac,
        RecordingSignalType::DelayProbe => {
            log::warn!(
                "DelayProbe selected in per-channel mode; use probe_channel_delays() instead. Falling back to Sweep."
            );
            SignalType::Sweep
        }
    }
}

/// Convert dBFS to amplitude without raising quiet requested levels.
///
/// Mirrors the engine's `measurement_amplitude_from_level_db` clamp added
/// with `prepare_measurement_signal`; frontends previously used an
/// unclamped `10^(level_db / 20)`.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "The clamped decibel level produces a bounded amplitude intentionally stored in f32."
)]
pub fn measurement_amplitude(level_db: f64) -> f32 {
    if !level_db.is_finite() {
        return 0.0;
    }
    10.0_f64.powf(level_db.min(0.0) / 20.0) as f32
}

/// Filesystem-safe file-name stem for a recording session or channel name.
///
/// Keeps alphanumerics, `_` and `-`; everything else (including path
/// separators) becomes `_`. An empty or all-unsafe input falls back to
/// `"recording"` so the result is always a usable file name.
#[must_use]
pub fn sanitize_recording_name(name: &str) -> String {
    let safe: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if safe.is_empty() {
        "recording".to_string()
    } else {
        safe
    }
}

/// Resolve the mic-calibration file for a capture: the per-mic entry at
/// `mic_index` wins, falling back to the session-global `global` path (B3).
///
/// `paths` is parallel to the recording channel mappings (indexed by mic
/// slot, **not** by hardware input channel). Empty strings are treated as
/// "no calibration". Returns the selected path, or `None` when neither the
/// per-mic slot nor the global path is set.
#[must_use]
pub fn resolve_mic_calibration(
    paths: &[Option<String>],
    mic_index: usize,
    global: Option<&str>,
) -> Option<String> {
    paths
        .get(mic_index)
        .and_then(std::clone::Clone::clone)
        .filter(|s| !s.is_empty())
        .or_else(|| global.filter(|s| !s.is_empty()).map(str::to_string))
}

/// Transfer-function average level (dB rel. unity) below which a completed
/// capture is flagged as suspiciously low (R10). Shared by all frontends.
pub const LOW_MEASURED_LEVEL_THRESHOLD_DB: f32 = -50.0;

/// Post-capture sanity check (R10): average the measured transfer-function
/// level (dB rel. unity) over the channel's own sweep band and return the
/// average when it falls below [`LOW_MEASURED_LEVEL_THRESHOLD_DB`].
///
/// The band is clamped to [20 Hz, 20 kHz]; bins at or below −150 dB are
/// treated as "no data" and ignored. The per-channel sweep band (not a
/// global band) is used so LFE channels with a narrow low-frequency sweep
/// are judged on their actual stimulus range. Returns `None` when the level
/// is healthy or when no usable bin falls inside the band.
///
/// Note this measures the *average measured level*, not a true acoustic
/// noise floor — phrase user-facing text accordingly (see
/// [`low_measured_level_warning`]).
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "Averaging f32 decibel values intentionally uses an f32 bin count."
)]
pub fn check_low_measured_level(
    frequencies: &[f32],
    level_db: &[f32],
    sweep_start_freq: f32,
    sweep_end_freq: f32,
) -> Option<f32> {
    let band_min = sweep_start_freq.max(20.0);
    let band_max = sweep_end_freq.min(20000.0);
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for (&freq, &mag) in frequencies.iter().zip(level_db.iter()) {
        if freq >= band_min && freq <= band_max && mag > -150.0 {
            sum += mag;
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let avg = sum / count as f32;
    (avg < LOW_MEASURED_LEVEL_THRESHOLD_DB).then_some(avg)
}

/// Canonical user-facing text for a [`check_low_measured_level`] hit,
/// worded honestly (low *measured level*, pointing at mic connection and
/// input gain — not a "noise floor"). `subjects` names the affected
/// channel(s) or speaker.
#[must_use]
pub fn low_measured_level_warning(subjects: &str) -> String {
    format!("Very low measured level on {subjects} — check mic connection and input gain")
}

/// Build the engine [`SignalParams`] for a per-channel capture (B1).
///
/// The sweep path routes through the engine's [`sweep_params_from_config`]
/// so the UI "Bass precision" knob (`bass_octave_duration_s`) and the
/// silence windows actually shape the generated stimulus (octave-scaled
/// sweep with more bass dwell) instead of only changing persisted metadata.
/// Pass `None` for `bass_octave_duration_s` to keep the legacy plain
/// fixed-duration log sweep.
///
/// `amp` should come from [`measurement_amplitude`] so the stimulus never
/// exceeds full scale. Noise/MLS/Dirac ignore the sweep-specific arguments.
#[must_use]
pub fn capture_signal_params(
    signal_type: SignalType,
    sweep_start_freq: f32,
    sweep_end_freq: f32,
    amp: f32,
    bass_octave_duration_s: Option<f32>,
    pre_silence_s: Option<f32>,
    post_silence_s: Option<f32>,
) -> SignalParams {
    match signal_type {
        SignalType::WhiteNoise | SignalType::PinkNoise | SignalType::MNoise => {
            SignalParams::Noise { amp }
        }
        SignalType::Mls => SignalParams::Mls {
            order: DEFAULT_MLS_ORDER,
            amp,
        },
        SignalType::Dirac => SignalParams::Dirac { amp },
        // Matches the frontends' historical catch-all: unexpected signal
        // types degrade to a sweep rather than failing the capture.
        _ => sweep_params_from_config(
            sweep_start_freq,
            sweep_end_freq,
            amp,
            bass_octave_duration_s,
            pre_silence_s,
            post_silence_s,
        ),
    }
}

// ============================================================================
// Per-take quality gate (Task 9, review §4 item 1)
// ============================================================================
//
// Verdict logic (thresholds, wording) lives here so the TUI and GPUI shells
// only render text and collect the user's accept/re-record choice. The
// engine's verdict is advisory: the capture succeeded, so the take is parked
// in `ChannelRecordingState::ReviewNeeded` until the user decides.

/// Highest sweep count the UIs offer for `num_sweeps`.
pub const MAX_NUM_SWEEPS: u16 = 8;

/// Clamp a user-entered sweep count to the UI range: 1 (single sweep) or
/// `MIN_REPEAT_SWEEPS..=MAX_NUM_SWEEPS`. **2 is never returned**: the engine
/// bumps it to `MIN_REPEAT_SWEEPS` anyway because two takes have zero
/// outlier-rejection power (commit 43d677d27), so offering it would lie
/// about what runs.
#[must_use]
pub fn clamp_num_sweeps(value: u16) -> u16 {
    match value.clamp(1, MAX_NUM_SWEEPS) {
        2 => MIN_REPEAT_SWEEPS,
        v => v,
    }
}

/// Step the sweep count by `delta` for +/- adjusters, skipping 2 in both
/// directions (see [`clamp_num_sweeps`]).
#[must_use]
pub fn nudge_num_sweeps(current: u16, delta: i32) -> u16 {
    let next = i32::from(current).saturating_add(delta);
    if next <= 1 {
        1
    } else if next == 2 {
        if delta > 0 { MIN_REPEAT_SWEEPS } else { 1 }
    } else {
        u16::try_from(next.min(i32::from(MAX_NUM_SWEEPS))).unwrap_or(MAX_NUM_SWEEPS)
    }
}

/// Extract the UI-facing quality summary from an engine [`CaptureAnalysis`]
/// (Task 7/8 wrapper). Pure field mapping — no decisions here.
///
/// `drift_ppm` stays an `Option`: `None` means the drift *estimation* was
/// unavailable (low-confidence end windows), which must never be rendered
/// as "0 ppm" (Task-7 review carry-forward).
#[must_use]
pub fn summarize_take_quality(capture: &CaptureAnalysis) -> TakeQualitySummary {
    TakeQualitySummary {
        trustworthy: capture.quality.trustworthy
            && capture
                .quality_findings
                .iter()
                .all(|finding| finding.status == QualityStatus::Pass),
        score: capture.quality.score,
        issues: capture.quality.issues.clone(),
        findings: capture.quality_findings.clone(),
        mean_coherence: capture.quality.mean_coherence,
        median_snr_db: capture.quality.median_snr_db,
        clip_fraction: capture.quality.clipping.fraction,
        drift_ppm: capture.drift.map(|d| d.ppm),
        drift_corrected: capture.drift_corrected,
        dropped_samples: capture.dropped_samples,
        accepted_count: capture.accepted_count,
        rejected_count: capture.rejected_count,
    }
}

/// One-line detailed verdict for a captured take — the issues plus every
/// available supporting metric (mean coherence, median SNR, clip fraction,
/// drift ppm, accepted/rejected takes). Shown when a take needs review and
/// in per-channel details.
///
/// Missing drift is rendered as `drift n/a`, never `0 ppm`.
pub fn take_verdict_text(q: &TakeQualitySummary) -> String {
    let mut parts: Vec<String> = Vec::new();
    if q.issues.is_empty() {
        parts.push(format!("score {:.2}", q.score));
    } else {
        parts.push(format!("score {:.2} — {}", q.score, q.issues.join("; ")));
    }
    if let Some(coherence) = q.mean_coherence {
        parts.push(format!("coherence {coherence:.2}"));
    }
    if let Some(snr) = q.median_snr_db {
        parts.push(format!("SNR {snr:.1} dB"));
    }
    if q.clip_fraction > 0.0 {
        parts.push(format!("clipped {:.2}%", q.clip_fraction * 100.0));
    }
    match q.drift_ppm {
        Some(ppm) => parts.push(format!(
            "drift {ppm:+.0} ppm{}",
            if q.drift_corrected {
                " (corrected)"
            } else {
                ""
            }
        )),
        None => parts.push("drift n/a".to_string()),
    }
    if q.rejected_count > 0 {
        parts.push(format!(
            "{}/{} takes accepted",
            q.accepted_count,
            q.accepted_count + q.rejected_count
        ));
    }
    if q.dropped_samples > 0 {
        parts.push(format!("dropped {} samples", q.dropped_samples));
    }
    let evidence = q
        .findings
        .iter()
        .filter(|finding| q.trustworthy || finding.status != QualityStatus::Pass)
        .map(format_quality_finding)
        .collect::<Vec<_>>();
    if !evidence.is_empty() {
        parts.push(format!("quality evidence: {}", evidence.join(" | ")));
    }
    parts.join(", ")
}

fn format_quality_finding(finding: &QualityFinding) -> String {
    let mut parts = vec![format!(
        "{} {}",
        quality_metric_label(finding.metric),
        match finding.status {
            QualityStatus::Pass => "pass",
            QualityStatus::Warning => "warning",
            QualityStatus::Unknown => "unknown",
        }
    )];
    if let Some(measured) = finding.measured {
        parts.push(format!(
            "{measured:.2} {}",
            quality_unit_label(finding.unit)
        ));
    }
    parts.push(quality_method_label(finding.method).to_string());
    if let Some(threshold) = finding.threshold {
        parts.push(format!(
            "threshold {} {:.2} {}",
            threshold_comparison_label(threshold.comparison),
            threshold.value,
            quality_unit_label(finding.unit)
        ));
    }
    if let Some(band) = finding.band {
        parts.push(format!("{:.0}–{:.0} Hz", band.low_hz, band.high_hz));
    }
    let mut text = format!("{} ({})", parts[0], parts[1..].join(", "));
    if let Some(reason) = finding.unknown_reason.as_deref() {
        let _ = write!(text, "; reason: {reason}");
    }
    if let Some(recommendation) = finding.recommendation.as_deref() {
        let _ = write!(text, "; next: {recommendation}");
    }
    text
}

fn quality_metric_label(metric: QualityMetric) -> &'static str {
    match metric {
        QualityMetric::TotalHarmonicDistortion => "THD",
        QualityMetric::BroadbandSnr => "broadband SNR",
        QualityMetric::BandSnr => "band SNR",
        QualityMetric::Clipping => "clipping",
        QualityMetric::TimingMarkers => "timing markers",
    }
}

fn quality_unit_label(unit: QualityUnit) -> &'static str {
    match unit {
        QualityUnit::Percent => "%",
        QualityUnit::Decibels => "dB",
        QualityUnit::Samples => "samples",
        QualityUnit::Count => "count",
    }
}

fn quality_method_label(method: QualityMethod) -> &'static str {
    match method {
        QualityMethod::FarinaHarmonicImpulseResponse => "Farina harmonic IR",
        QualityMethod::BroadbandPowerRatio => "active/silent-window power ratio",
        QualityMethod::NoiseSubtractedBandPowerRatio => "noise-subtracted band power ratio",
        QualityMethod::FullScaleSampleCount => "full-scale sample count",
        QualityMethod::TimingMarkerFit => "timing-marker fit",
    }
}

fn threshold_comparison_label(comparison: ThresholdComparison) -> &'static str {
    match comparison {
        ThresholdComparison::AtMost => "≤",
        ThresholdComparison::AtLeast => "≥",
        ThresholdComparison::EqualTo => "=",
    }
}

/// Compact per-channel quality cell for channel-list tables.
///
/// - `REVIEW <score>` — take is parked awaiting the user's decision;
/// - `OK* <score>` — accepted despite quality warnings (distinct from clean);
/// - `OK <score>` — clean take;
/// - empty string when there is nothing to say yet.
#[must_use]
pub fn take_quality_cell(state: ChannelRecordingState, result: Option<&RecordingResult>) -> String {
    let quality = result.and_then(|r| r.quality.as_ref());
    match (state, quality) {
        (ChannelRecordingState::ReviewNeeded, Some(q)) => format!("REVIEW {:.2}", q.score),
        // Accepted-with-warning: deliberately distinct from a clean Done.
        (ChannelRecordingState::Done, Some(q)) if !q.trustworthy => {
            format!("OK* {:.2}", q.score)
        }
        (ChannelRecordingState::Done, Some(q)) => format!("OK {:.2}", q.score),
        // Done/ReviewNeeded without a quality summary means a legacy or
        // loaded session — say so rather than implying a verdict.
        (ChannelRecordingState::Done | ChannelRecordingState::ReviewNeeded, None) => {
            "no data".to_string()
        }
        _ => String::new(),
    }
}

/// Dropout warning (R6 / §4 item 1): `Some` text when the capture dropped
/// input samples to ring-buffer overruns.
#[must_use]
pub fn dropout_warning(dropped_samples: u64) -> Option<String> {
    (dropped_samples > 0).then(|| {
        format!(
            "{dropped_samples} samples dropped during capture — USB/driver underrun; \
             results may be unreliable, consider re-measuring"
        )
    })
}

/// Multi-position guidance (Phase 3 item 10): one-line hint for the capture
/// flow when `num_positions > 1`. `pos` is the 0-based position about to be
/// recorded. The first position anchors delays/levels, so it must be the
/// main listening position; later positions should stay within roughly 60 cm
/// of it (Audyssey/Dirac guidance).
#[must_use]
pub fn position_guidance(pos: usize, total: usize) -> String {
    if pos == 0 {
        format!("Position 1 of {total}: place the mic(s) at the main listening position first")
    } else {
        format!(
            "Move the mic(s) to position {} of {total} — stay within ~60 cm of the main listening position",
            pos + 1
        )
    }
}

/// Severity of one [`session_quality_lines`] entry — the structured form of
/// the summary line, so UIs color by kind instead of substring-matching the
/// rendered English text (task-9 review B3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionQualityLineKind {
    /// Clean take (trustworthy).
    Ok,
    /// Take with quality warnings (parked for review or accepted-with-warning).
    Review,
    /// Recorded channel without quality data (legacy / loaded session).
    NoData,
}

/// Session quality summary (§4 item 1 / Phase 2 item 5): one line per
/// recorded channel with its score and warnings, so the user can see which
/// positions to re-measure before leaving the screen. Channels are listed in
/// recording order (position-major), which matches the on-screen channel
/// list.
///
/// Only `Done` and `ReviewNeeded` channels are listed (task-9 review B2): a
/// channel whose retake FAILED keeps its previous result in memory, and
/// listing that stale quality would misrepresent the session.
#[must_use]
pub fn session_quality_lines(
    channels: &[ChannelRecording],
) -> Vec<(String, SessionQualityLineKind)> {
    channels
        .iter()
        .filter(|c| {
            matches!(
                c.state,
                ChannelRecordingState::Done | ChannelRecordingState::ReviewNeeded
            ) && c.result.is_some()
        })
        .map(|c| {
            let quality = c.result.as_ref().and_then(|r| r.quality.as_ref());
            let (mut line, kind) = match quality {
                Some(q) if q.trustworthy => (
                    format!(
                        "{}: OK ({}/{} sweeps) — {}",
                        c.channel_name,
                        q.accepted_count,
                        q.accepted_count + q.rejected_count,
                        take_verdict_text(q),
                    ),
                    SessionQualityLineKind::Ok,
                ),
                Some(q) => (
                    format!(
                        "{}: REVIEW (score {:.2}) — {}",
                        c.channel_name,
                        q.score,
                        take_verdict_text(q)
                    ),
                    SessionQualityLineKind::Review,
                ),
                None => (
                    format!("{}: recorded (no quality data)", c.channel_name),
                    SessionQualityLineKind::NoData,
                ),
            };
            if let Some(q) = quality
                && q.dropped_samples > 0
            {
                let _ = write!(line, "; {} dropped samples", q.dropped_samples);
            }
            (line, kind)
        })
        .collect()
}

/// Text-only view of [`session_quality_lines`] for plain-text UIs.
#[must_use]
pub fn session_quality_summary(channels: &[ChannelRecording]) -> Vec<String> {
    session_quality_lines(channels)
        .into_iter()
        .map(|(line, _)| line)
        .collect()
}

/// Truthful `num_sweeps` to persist in `RecordingConfiguration` (Task 8
/// metadata rule): the **minimum engine-reported accepted-take count** over
/// completed channels (one bad channel must not be covered up by the
/// requested count). `None` when no completed channel carries quality data
/// (legacy / loaded sessions), preserving the old "unknown" semantics
/// instead of persisting the requested count as if it were measured.
#[must_use]
pub fn accepted_num_sweeps_for_save(channels: &[ChannelRecording]) -> Option<u8> {
    let mut min: Option<usize> = None;
    for c in channels {
        if c.state != ChannelRecordingState::Done {
            continue;
        }
        if let Some(q) = c.result.as_ref().and_then(|r| r.quality.as_ref()) {
            min = Some(min.map_or(q.accepted_count, |m: usize| m.min(q.accepted_count)));
        }
    }
    min.map(|m| u8::try_from(m).unwrap_or(u8::MAX))
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "Synthetic fixtures use bounded sample counts and intentionally quantized audio; exact comparisons verify unchanged settings and samples."
)]
mod tests {
    use super::*;

    #[test]
    fn enrich_results_from_take_csvs_restores_advanced_metrics() {
        let directory = tempfile::tempdir().unwrap();
        let csv_path = directory.path().join("L.csv");
        std::fs::write(
            &csv_path,
            "frequency_hz,spl_db,phase_deg,thd_percent,rt60_ms,c50_db,c80_db,group_delay_ms\n\
             100.000000,-10.000,0.000000,1.500000,320.000,12.000,15.000,0.500000\n\
             1000.000000,-9.000,0.000000,NaN,310.000,13.000,16.000,0.400000\n",
        )
        .unwrap();

        let fr_only = |frequencies: Vec<f32>| RecordingResult {
            channel: 0,
            sample_rate_hz: Some(48_000),
            wav_path: None,
            csv_path: Some(csv_path.to_string_lossy().into_owned()),
            frequencies,
            magnitude_db: vec![-10.0, -9.0],
            phase_deg: vec![0.0, 0.0],
            impulse_response: None,
            impulse_time_ms: None,
            thd_percent: None,
            harmonic_distortion_db: None,
            excess_group_delay_ms: None,
            rt60_ms: None,
            clarity_c50_db: None,
            clarity_c80_db: None,
            spectrogram_db: None,
            quality: None,
            stimulus_config: None,
        };
        let mut take = ChannelRecording::new(0, "L".to_string());
        take.result = Some(fr_only(vec![100.0, 1000.0]));
        let mut recordings = vec![take];
        enrich_results_from_take_csvs(&mut recordings);
        let result = recordings[0].result.as_ref().unwrap();
        assert_eq!(result.thd_percent.as_ref().unwrap()[0], 1.5);
        assert!(result.thd_percent.as_ref().unwrap()[1].is_nan());
        let json = serde_json::to_value(result).unwrap();
        assert!(json["thd_percent"][1].is_null());
        let restored: RecordingResult = serde_json::from_value(json).unwrap();
        assert!(restored.thd_percent.unwrap()[1].is_nan());
        assert_eq!(result.rt60_ms.as_deref(), Some(&[320.0, 310.0][..]));
        assert_eq!(result.clarity_c50_db.as_deref(), Some(&[12.0, 13.0][..]));
        assert_eq!(
            result.excess_group_delay_ms.as_deref(),
            Some(&[0.5, 0.4][..])
        );

        // Existing THD and RT60 do not suppress missing clarity or group delay.
        let mut partial = ChannelRecording::new(0, "L".into());
        let mut existing = fr_only(vec![100.0, 1000.0]);
        existing.thd_percent = Some(vec![0.0, 2.0]);
        existing.rt60_ms = Some(vec![111.0, 222.0]);
        partial.result = Some(existing);
        let mut partial = vec![partial];
        enrich_results_from_take_csvs(&mut partial);
        let result = partial[0].result.as_ref().unwrap();
        assert_eq!(result.thd_percent.as_deref(), Some(&[0.0, 2.0][..]));
        assert_eq!(result.rt60_ms.as_deref(), Some(&[111.0, 222.0][..]));
        assert_eq!(result.clarity_c50_db.as_deref(), Some(&[12.0, 13.0][..]));
        assert!(result.excess_group_delay_ms.is_some());

        let mut invalid = ChannelRecording::new(0, "L".into());
        invalid.result = Some(fr_only(vec![100.0, f32::NAN]));
        let mut invalid = vec![invalid];
        enrich_results_from_take_csvs(&mut invalid);
        assert!(invalid[0].result.as_ref().unwrap().thd_percent.is_none());

        // A mismatched frequency grid must not be enriched.
        let mut take = ChannelRecording::new(0, "L".to_string());
        take.result = Some(fr_only(vec![100.0, 2000.0]));
        let mut recordings = vec![take];
        enrich_results_from_take_csvs(&mut recordings);
        assert!(recordings[0].result.as_ref().unwrap().thd_percent.is_none());
    }

    #[test]
    fn session_save_replaces_complete_metadata_and_cleans_temporary_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(RECORDINGS_FILENAME);
        save_recording_session_json(&path, br#"{"revision":1}"#).unwrap();
        save_recording_session_json(&path, br#"{"revision":2}"#).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), br#"{"revision":2}"#);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn session_save_preserves_previous_metadata_after_partial_write_failure() {
        use std::io::Write;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(RECORDINGS_FILENAME);
        let original = br#"{"revision":1}"#;
        std::fs::write(&path, original).unwrap();
        let error = replace_recording_session(&path, |file| {
            file.write_all(b"incomplete replacement")?;
            Err(std::io::Error::other("simulated full disk"))
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "simulated full disk");
        assert_eq!(std::fs::read(&path).unwrap(), original);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        save_recording_session_json(&path, br#"{"revision":2}"#).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), br#"{"revision":2}"#);
    }

    #[test]
    fn signal_type_for_maps_each_variant() {
        assert_eq!(
            signal_type_for(RecordingSignalType::Sweep),
            SignalType::Sweep
        );
        assert_eq!(
            signal_type_for(RecordingSignalType::WhiteNoise),
            SignalType::WhiteNoise
        );
        assert_eq!(
            signal_type_for(RecordingSignalType::PinkNoise),
            SignalType::PinkNoise
        );
        assert_eq!(signal_type_for(RecordingSignalType::Mls), SignalType::Mls);
        assert_eq!(
            signal_type_for(RecordingSignalType::Dirac),
            SignalType::Dirac
        );
    }

    #[test]
    fn signal_type_for_delay_probe_falls_back_to_sweep() {
        assert_eq!(
            signal_type_for(RecordingSignalType::DelayProbe),
            SignalType::Sweep
        );
    }

    #[test]
    fn measurement_amplitude_converts_db() {
        assert!((measurement_amplitude(-6.0206) - 0.5).abs() < 1e-4);
        assert_eq!(measurement_amplitude(0.0), 1.0);
        assert!((measurement_amplitude(-40.0) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn measurement_amplitude_clamps_to_full_scale() {
        // Positive dB must never exceed 0 dBFS.
        assert_eq!(measurement_amplitude(6.0), 1.0);
        assert_eq!(measurement_amplitude(20.0), 1.0);
        assert!((measurement_amplitude(-80.0) - 0.0001).abs() < 1e-8);
        assert!((measurement_amplitude(-60.0) - 0.001).abs() < 1e-8);
        assert_eq!(measurement_amplitude(f64::NAN), 0.0);
    }

    #[test]
    fn sanitize_recording_name_replaces_unsafe_chars() {
        assert_eq!(sanitize_recording_name("My Recording!"), "My_Recording_");
        assert_eq!(sanitize_recording_name("a/b\\c:d"), "a_b_c_d");
        assert_eq!(sanitize_recording_name("LFE-1_ok"), "LFE-1_ok");
        assert_eq!(sanitize_recording_name("  padded  "), "padded");
        // Unicode alphanumerics are kept.
        assert_eq!(sanitize_recording_name("café"), "café");
    }

    #[test]
    fn sanitize_recording_name_empty_falls_back() {
        assert_eq!(sanitize_recording_name(""), "recording");
        assert_eq!(sanitize_recording_name("   "), "recording");
        assert_eq!(sanitize_recording_name("!!!"), "___");
    }

    #[test]
    fn resolve_mic_calibration_prefers_per_mic_slot() {
        let paths = vec![Some("mic0.txt".to_string()), Some("mic1.txt".to_string())];
        assert_eq!(
            resolve_mic_calibration(&paths, 1, Some("global.txt")),
            Some("mic1.txt".to_string())
        );
    }

    #[test]
    fn resolve_mic_calibration_falls_back_to_global() {
        // Missing slot.
        let paths = vec![None, Some("mic1.txt".to_string())];
        assert_eq!(
            resolve_mic_calibration(&paths, 0, Some("global.txt")),
            Some("global.txt".to_string())
        );
        // Out-of-range index.
        assert_eq!(
            resolve_mic_calibration(&paths, 7, Some("global.txt")),
            Some("global.txt".to_string())
        );
        // Empty vec.
        assert_eq!(
            resolve_mic_calibration(&[], 0, Some("global.txt")),
            Some("global.txt".to_string())
        );
    }

    #[test]
    fn resolve_mic_calibration_treats_empty_strings_as_unset() {
        let paths = vec![Some(String::new())];
        assert_eq!(
            resolve_mic_calibration(&paths, 0, Some("global.txt")),
            Some("global.txt".to_string())
        );
        // Empty global is also filtered out.
        assert_eq!(resolve_mic_calibration(&paths, 0, Some("")), None);
        assert_eq!(resolve_mic_calibration(&[], 0, None), None);
    }

    #[test]
    fn check_low_measured_level_flags_low_band_average() {
        // Flat -80 dB across the band → flagged, average returned.
        let freqs: Vec<f32> = (0..100).map(|i| 20.0 + i as f32 * 200.0).collect();
        let levels = vec![-80.0; 100];
        let avg = check_low_measured_level(&freqs, &levels, 20.0, 20000.0)
            .expect("low band average should be flagged");
        assert!((avg - -80.0).abs() < 1e-4);
    }

    #[test]
    fn check_low_measured_level_passes_healthy_level() {
        let freqs: Vec<f32> = (0..100).map(|i| 20.0 + i as f32 * 200.0).collect();
        let levels = vec![-10.0; 100];
        assert!(check_low_measured_level(&freqs, &levels, 20.0, 20000.0).is_none());
        // Exactly at the threshold is not "below".
        let levels = vec![LOW_MEASURED_LEVEL_THRESHOLD_DB; 100];
        assert!(check_low_measured_level(&freqs, &levels, 20.0, 20000.0).is_none());
    }

    #[test]
    fn check_low_measured_level_restricts_to_sweep_band() {
        // Low level outside the sweep band must not trip the check (the
        // per-channel band semantics matter for LFE channels).
        let freqs = vec![5.0, 100.0, 1000.0];
        let levels = vec![-90.0, -10.0, -10.0];
        assert!(check_low_measured_level(&freqs, &levels, 20.0, 500.0).is_none());
        // Low level inside a narrow LFE band is caught.
        let levels = vec![-90.0, -80.0, -10.0];
        assert!(check_low_measured_level(&freqs, &levels, 20.0, 500.0).is_some());
    }

    #[test]
    fn check_low_measured_level_ignores_no_data_bins() {
        // Bins at or below -150 dB carry no information and are skipped.
        let freqs = vec![100.0, 200.0, 300.0];
        let levels = vec![-150.0, -200.0, -10.0];
        assert!(check_low_measured_level(&freqs, &levels, 20.0, 20000.0).is_none());
        // All bins unusable → no verdict rather than a false alarm.
        let levels = vec![-150.0, -200.0, -160.0];
        assert!(check_low_measured_level(&freqs, &levels, 20.0, 20000.0).is_none());
        assert!(check_low_measured_level(&[], &[], 20.0, 20000.0).is_none());
    }

    #[test]
    fn low_measured_level_warning_names_subjects() {
        let msg = low_measured_level_warning("FL, FR");
        assert!(msg.contains("Very low measured level"));
        assert!(msg.contains("FL, FR"));
        assert!(msg.contains("mic connection"));
    }

    #[test]
    fn capture_signal_params_sweep_uses_octave_sweep_when_configured() {
        let params = capture_signal_params(
            SignalType::Sweep,
            20.0,
            20000.0,
            0.5,
            Some(3.0),
            Some(2.0),
            None,
        );
        match params {
            SignalParams::OctaveSweep {
                start_freq,
                end_freq,
                amp,
                bass_octave_duration_s,
                pre_silence_s,
                post_silence_s,
            } => {
                assert_eq!(start_freq, 20.0);
                assert_eq!(end_freq, 20000.0);
                assert_eq!(amp, 0.5);
                assert_eq!(bass_octave_duration_s, 3.0);
                assert_eq!(pre_silence_s, 2.0);
                // Engine default applies when the UI leaves it unset.
                assert_eq!(post_silence_s, 2.0);
            }
            other => panic!("expected OctaveSweep, got {other:?}"),
        }
    }

    #[test]
    fn capture_signal_params_sweep_without_bass_duration_stays_legacy() {
        let params = capture_signal_params(SignalType::Sweep, 20.0, 20000.0, 0.5, None, None, None);
        assert!(matches!(params, SignalParams::Sweep { .. }));
    }

    #[test]
    fn capture_signal_params_other_types_ignore_sweep_args() {
        assert!(matches!(
            capture_signal_params(
                SignalType::PinkNoise,
                20.0,
                20000.0,
                0.25,
                Some(3.0),
                Some(2.0),
                None
            ),
            SignalParams::Noise { amp } if amp == 0.25
        ));
        assert!(matches!(
            capture_signal_params(SignalType::Mls, 20.0, 20000.0, 0.25, None, None, None),
            SignalParams::Mls {
                order: DEFAULT_MLS_ORDER,
                amp
            } if amp == 0.25
        ));
        assert!(matches!(
            capture_signal_params(SignalType::Dirac, 20.0, 20000.0, 0.25, None, None, None),
            SignalParams::Dirac { amp } if amp == 0.25
        ));
    }

    // === Task 9: num_sweeps range, quality verdict text, session summary ===

    #[test]
    fn clamp_num_sweeps_never_returns_two() {
        assert_eq!(clamp_num_sweeps(0), 1);
        assert_eq!(clamp_num_sweeps(1), 1);
        // 2 has zero outlier-rejection power; the engine bumps it to 3.
        assert_eq!(clamp_num_sweeps(2), MIN_REPEAT_SWEEPS);
        assert_eq!(clamp_num_sweeps(3), 3);
        assert_eq!(clamp_num_sweeps(8), 8);
        assert_eq!(clamp_num_sweeps(9), MAX_NUM_SWEEPS);
        assert_eq!(clamp_num_sweeps(100), MAX_NUM_SWEEPS);
    }

    #[test]
    fn nudge_num_sweeps_skips_two_in_both_directions() {
        assert_eq!(nudge_num_sweeps(1, 1), 3);
        assert_eq!(nudge_num_sweeps(1, -1), 1);
        assert_eq!(nudge_num_sweeps(3, -1), 1);
        assert_eq!(nudge_num_sweeps(3, 1), 4);
        assert_eq!(nudge_num_sweeps(8, 1), 8);
        assert_eq!(nudge_num_sweeps(4, -1), 3);
    }

    fn quality_summary(
        trustworthy: bool,
        issues: &[&str],
        drift_ppm: Option<f64>,
    ) -> TakeQualitySummary {
        TakeQualitySummary {
            trustworthy,
            score: 0.42,
            issues: issues
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            findings: Vec::new(),
            mean_coherence: Some(0.71),
            median_snr_db: Some(18.2),
            clip_fraction: 0.003,
            drift_ppm,
            drift_corrected: false,
            dropped_samples: 0,
            accepted_count: 4,
            rejected_count: 1,
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "This end-to-end fixture keeps setup, execution, and artifact assertions together."
    )]
    fn recording_quality_sidecar_roundtrips_warning_and_unknown_by_source() {
        use crate::quality::{
            QualityBand, QualityFinding, QualityMethod, QualityMetric, QualityPolicy,
            QualityStatus, QualityThreshold, QualityUnit, ThresholdComparison,
        };

        let directory = tempfile::tempdir().unwrap();
        let session_path = directory.path().join(RECORDINGS_FILENAME);
        let session_bytes = br#"{"speakers":{"Left":{},"Right":{}}}"#;
        save_recording_session_json(&session_path, session_bytes).unwrap();
        let warning_source = RecordingSourceIdentity {
            speaker: "Left".into(),
            driver_index: None,
            measurement_index: 0,
        };
        let unknown_source = RecordingSourceIdentity {
            speaker: "Left".into(),
            driver_index: None,
            measurement_index: 1,
        };
        let adjacent_source = RecordingSourceIdentity {
            speaker: "Right".into(),
            driver_index: None,
            measurement_index: 0,
        };
        let mut warning = quality_summary(false, &["high harmonic distortion"], None);
        warning.findings.push(QualityFinding {
            policy: QualityPolicy::SweepAnalysis,
            metric: QualityMetric::TotalHarmonicDistortion,
            status: QualityStatus::Warning,
            measured: Some(12.5),
            unit: QualityUnit::Percent,
            band: Some(QualityBand {
                low_hz: 100.0,
                high_hz: 20_000.0,
            }),
            method: QualityMethod::FarinaHarmonicImpulseResponse,
            threshold: Some(QualityThreshold {
                comparison: ThresholdComparison::AtMost,
                value: 10.0,
            }),
            unknown_reason: None,
            recommendation: Some("lower playback level and repeat".into()),
            message: "THD exceeds advisory threshold".into(),
        });
        let mut unknown = quality_summary(false, &["SNR evidence unavailable"], None);
        unknown
            .findings
            .push(QualityFinding::unknown_with_threshold(
                QualityPolicy::CoherentCapture,
                QualityMetric::BroadbandSnr,
                QualityUnit::Decibels,
                QualityMethod::BroadbandPowerRatio,
                QualityThreshold {
                    comparison: ThresholdComparison::AtLeast,
                    value: 30.0,
                },
                "the quiet window is digital silence",
                Some("capture a valid quiet interval".into()),
                "coherent SNR is unavailable",
            ));

        let warning_stimulus = effective_acquisition_config(
            SignalType::Sweep,
            &capture_signal_params(
                SignalType::Sweep,
                20.0,
                20_000.0,
                0.5,
                Some(3.0),
                Some(2.0),
                Some(1.0),
            ),
            5.0,
            48_000,
        )
        .unwrap();
        let unknown_stimulus = effective_acquisition_config(
            SignalType::Sweep,
            &capture_signal_params(
                SignalType::Sweep,
                20.0,
                20_000.0,
                0.1,
                Some(3.0),
                Some(2.0),
                Some(1.0),
            ),
            5.0,
            48_000,
        )
        .unwrap();

        save_recording_quality_sidecar_with_stimuli(
            &session_path,
            session_bytes,
            vec![
                RecordingQualityEntry {
                    source: warning_source.clone(),
                    quality: warning.clone(),
                },
                RecordingQualityEntry {
                    source: unknown_source.clone(),
                    quality: unknown.clone(),
                },
            ],
            vec![
                RecordingStimulusEntry {
                    source: warning_source.clone(),
                    config: warning_stimulus.clone(),
                },
                RecordingStimulusEntry {
                    source: unknown_source.clone(),
                    config: unknown_stimulus.clone(),
                },
                RecordingStimulusEntry {
                    source: adjacent_source.clone(),
                    config: warning_stimulus.clone(),
                },
            ],
        )
        .unwrap();

        let loaded = load_recording_quality_sidecar(&session_path, session_bytes)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.quality_for(&warning_source), Some(&warning));
        assert_eq!(loaded.quality_for(&unknown_source), Some(&unknown));
        assert_eq!(loaded.quality_for(&adjacent_source), None);
        assert_eq!(
            loaded.stimulus_for(&warning_source),
            Some(&warning_stimulus)
        );
        assert_eq!(
            loaded.stimulus_for(&unknown_source),
            Some(&unknown_stimulus)
        );
        assert_eq!(
            loaded.stimulus_for(&adjacent_source),
            Some(&warning_stimulus)
        );
        assert_eq!(
            loaded.entries.len(),
            2,
            "neighboring source is retained separately"
        );
        let changed_session = br#"{"updated":true}"#;
        save_recording_session_json(&session_path, changed_session).unwrap();
        assert_eq!(
            load_recording_quality_sidecar(&session_path, br#"{"updated":true}"#).unwrap(),
            None,
            "stale quality must not attach after the canonical session changes"
        );

        let legacy_directory = tempfile::tempdir().unwrap();
        let missing_session = legacy_directory.path().join(RECORDINGS_FILENAME);
        save_recording_session_json(&missing_session, session_bytes).unwrap();
        assert_eq!(
            load_recording_quality_sidecar(&missing_session, session_bytes).unwrap(),
            None,
            "legacy sessions without a sidecar retain no inferred quality"
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "This end-to-end fixture keeps setup, execution, and artifact assertions together."
    )]
    fn quality_entries_follow_saved_speaker_mic_position_order_without_shifting_missing_quality() {
        let result = |channel, name: &str, quality| RecordingResult {
            channel,
            sample_rate_hz: Some(48_000),
            wav_path: Some(format!("{name}.wav")),
            csv_path: Some(format!("{name}.csv")),
            frequencies: vec![100.0],
            magnitude_db: vec![0.0],
            phase_deg: vec![0.0],
            impulse_response: None,
            impulse_time_ms: None,
            thd_percent: None,
            harmonic_distortion_db: None,
            excess_group_delay_ms: None,
            rt60_ms: None,
            clarity_c50_db: None,
            clarity_c80_db: None,
            spectrogram_db: None,
            quality,
            stimulus_config: None,
        };
        let make_take =
            |channel_index, channel_name: &str, mic_index, mic_position_index, quality| {
                let mut take = ChannelRecording::with_mic_position(
                    channel_index,
                    channel_name.into(),
                    mic_index,
                    mic_position_index,
                );
                take.state = ChannelRecordingState::Done;
                take.result = Some(result(channel_index, channel_name, quality));
                take
            };

        // Save order is position first, then mic, matching the RoomConfig
        // builder. The first source deliberately has no quality evidence.
        let recordings = vec![
            make_take(0, "L (Pos 1 / Mic 1)", 0, 0, None),
            make_take(
                0,
                "L (Pos 1 / Mic 2)",
                1,
                0,
                Some(quality_summary(false, &["high THD"], None)),
            ),
            make_take(
                0,
                "L (Pos 2 / Mic 1)",
                0,
                1,
                Some(quality_summary(false, &["SNR unavailable"], None)),
            ),
            make_take(
                1,
                "R (Pos 1 / Mic 1)",
                0,
                0,
                Some(quality_summary(true, &[], Some(0.0))),
            ),
        ];
        let mut recordings = recordings;
        let stimulus = |amplitude| {
            effective_acquisition_config(
                SignalType::Sweep,
                &capture_signal_params(
                    SignalType::Sweep,
                    20.0,
                    20_000.0,
                    amplitude,
                    Some(3.0),
                    Some(2.0),
                    Some(1.0),
                ),
                5.0,
                48_000,
            )
            .unwrap()
        };
        let old_left_config = stimulus(0.5);
        let retaken_left_config = stimulus(0.25);
        let right_config = stimulus(0.5);
        recordings[1].result.as_mut().unwrap().stimulus_config = Some(old_left_config.clone());
        recordings[2].result.as_mut().unwrap().stimulus_config = Some(retaken_left_config.clone());
        recordings[3].result.as_mut().unwrap().stimulus_config = Some(right_config.clone());
        let entries = recording_quality_entries_for_sources(&recordings, &["L".into(), "R".into()]);
        let stimulus_entries =
            recording_stimulus_entries_from_sources(&recordings, &["L".into(), "R".into()]);

        let identity = |speaker: &str, measurement_index| RecordingSourceIdentity {
            speaker: speaker.into(),
            driver_index: None,
            measurement_index,
        };
        let sidecar = RecordingQualitySidecar {
            version: RECORDING_QUALITY_SIDECAR_VERSION,
            session_sha256: String::new(),
            session_modified_unix_nanos: 0,
            entries,
            stimuli: Vec::new(),
        };
        assert_eq!(sidecar.quality_for(&identity("L", 0)), None);
        assert_eq!(
            sidecar.quality_for(&identity("L", 1)).unwrap().issues,
            vec!["high THD"]
        );
        assert_eq!(
            sidecar.quality_for(&identity("L", 2)).unwrap().issues,
            vec!["SNR unavailable"]
        );
        assert_eq!(
            sidecar.quality_for(&identity("R", 0)).unwrap().drift_ppm,
            Some(0.0)
        );
        assert_eq!(
            sidecar.stimuli,
            [] as [crate::recording_helpers::RecordingStimulusEntry; 0]
        );
        assert_eq!(stimulus_entries.len(), 3);
        assert_eq!(
            stimulus_entries[0].source,
            identity("L", 1),
            "missing first-mic evidence must not shift source identity"
        );
        assert_eq!(stimulus_entries[0].config, old_left_config);
        assert_eq!(stimulus_entries[1].config, retaken_left_config);
        assert_eq!(stimulus_entries[2].config, right_config);

        let mut imported_again = recordings.clone();
        for recording in &mut imported_again {
            if let Some(result) = recording.result.as_mut() {
                result.stimulus_config = None;
            }
        }
        let roundtripped = RecordingQualitySidecar {
            version: RECORDING_QUALITY_SIDECAR_VERSION,
            session_sha256: String::new(),
            session_modified_unix_nanos: 0,
            entries: Vec::new(),
            stimuli: stimulus_entries,
        };
        restore_recording_stimuli_from_sidecar(
            &mut imported_again,
            &["L".into(), "R".into()],
            &roundtripped,
        );
        assert_eq!(
            imported_again[1].result.as_ref().unwrap().stimulus_config,
            Some(old_left_config)
        );
        assert_eq!(
            imported_again[2].result.as_ref().unwrap().stimulus_config,
            Some(retaken_left_config)
        );
    }

    #[test]
    fn effective_acquisition_duration_matches_generated_playback_frames() {
        let sample_rate = 48_000;
        let params = capture_signal_params(
            SignalType::Sweep,
            20.0,
            20_000.0,
            0.5,
            Some(3.0),
            Some(2.0),
            Some(1.0),
        );
        let effective =
            effective_acquisition_config(SignalType::Sweep, &params, 5.0, sample_rate).unwrap();
        let generated = crate::signal_recorder::prepare_measurement_signal(
            SignalType::Sweep,
            &params,
            5.0,
            sample_rate,
        )
        .unwrap();
        assert!(!effective.duration_control_applies);
        assert_eq!(effective.playback_frames, generated.len());
        assert_eq!(
            effective.playback_duration_secs,
            generated.len() as f32 / sample_rate as f32
        );

        let different_nominal_duration =
            effective_acquisition_config(SignalType::Sweep, &params, 12.0, sample_rate).unwrap();
        assert_eq!(
            effective.playback_frames,
            different_nominal_duration.playback_frames
        );
        assert_ne!(
            effective.requested_signal_duration_secs,
            different_nominal_duration.requested_signal_duration_secs
        );

        let slower_params = capture_signal_params(
            SignalType::Sweep,
            20.0,
            20_000.0,
            0.5,
            Some(5.0),
            Some(2.0),
            Some(1.0),
        );
        let slower =
            effective_acquisition_config(SignalType::Sweep, &slower_params, 5.0, sample_rate)
                .unwrap();
        assert!(slower.playback_frames > effective.playback_frames);

        let legacy_params =
            capture_signal_params(SignalType::Sweep, 20.0, 20_000.0, 0.5, None, None, None);
        let legacy =
            effective_acquisition_config(SignalType::Sweep, &legacy_params, 2.25, sample_rate)
                .unwrap();
        assert!(legacy.duration_control_applies);
        assert_eq!(legacy.generated_frames, 108_000);
    }

    #[test]
    fn effective_duration_marks_generic_controls_that_are_ignored() {
        let sample_rate = 48_000;
        let tone = effective_acquisition_config(
            SignalType::Tone,
            &SignalParams::Tone {
                freq: 1_000.0,
                amp: 0.5,
            },
            1.5,
            sample_rate,
        )
        .unwrap();
        assert!(tone.duration_control_applies);
        assert_eq!(tone.generated_frames, 72_000);

        let noise = effective_acquisition_config(
            SignalType::PinkNoise,
            &SignalParams::Noise { amp: 0.25 },
            2.0,
            sample_rate,
        )
        .unwrap();
        assert!(noise.duration_control_applies);
        assert_eq!(noise.generated_frames, 96_000);

        let mls = effective_acquisition_config(
            SignalType::Mls,
            &SignalParams::Mls { order: 8, amp: 0.5 },
            0.0,
            sample_rate,
        )
        .unwrap();
        assert!(!mls.duration_control_applies);
        assert_eq!(mls.generated_frames, 255);
    }

    #[test]
    fn recording_stimulus_upsert_keeps_neighboring_source_provenance() {
        let source = |speaker: &str| RecordingSourceIdentity {
            speaker: speaker.to_string(),
            driver_index: None,
            measurement_index: 0,
        };
        let config = |amplitude_peak| EffectiveAcquisitionConfig {
            signal_type: "sweep".into(),
            sample_rate_hz: 48_000,
            requested_signal_duration_secs: 5.0,
            duration_control_applies: false,
            generated_frames: 480_000,
            generated_duration_secs: 10.0,
            playback_padding_frames: 24_000,
            playback_frames: 504_000,
            playback_duration_secs: 10.5,
            amplitude_peak: Some(amplitude_peak),
            level_dbfs: Some(20.0 * amplitude_peak.log10()),
            low_frequency_hz: Some(20.0),
            high_frequency_hz: Some(20_000.0),
            bass_octave_duration_secs: Some(3.0),
            pre_silence_secs: Some(2.0),
            post_silence_secs: Some(1.0),
        };
        let mut sidecar = RecordingQualitySidecar {
            version: RECORDING_QUALITY_SIDECAR_VERSION,
            session_sha256: String::new(),
            session_modified_unix_nanos: 0,
            entries: Vec::new(),
            stimuli: vec![RecordingStimulusEntry {
                source: source("L"),
                config: config(0.5),
            }],
        };
        sidecar.upsert_stimulus(RecordingStimulusEntry {
            source: source("L"),
            config: config(0.25),
        });
        sidecar.upsert_stimulus(RecordingStimulusEntry {
            source: source("R"),
            config: config(0.5),
        });
        assert_eq!(sidecar.stimuli.len(), 2);
        assert_eq!(
            sidecar.stimulus_for(&source("L")).unwrap().amplitude_peak,
            Some(0.25)
        );
        assert_eq!(
            sidecar.stimulus_for(&source("R")).unwrap().amplitude_peak,
            Some(0.5)
        );
    }

    #[test]
    fn take_verdict_text_lists_issues_and_metrics() {
        let q = quality_summary(false, &["clipping detected"], Some(-45.0));
        let text = take_verdict_text(&q);
        assert!(text.contains("score 0.42"), "{text}");
        assert!(text.contains("clipping detected"), "{text}");
        assert!(text.contains("coherence 0.71"), "{text}");
        assert!(text.contains("SNR 18.2 dB"), "{text}");
        assert!(text.contains("clipped 0.30%"), "{text}");
        assert!(text.contains("drift -45 ppm"), "{text}");
        assert!(text.contains("4/5 takes accepted"), "{text}");
        assert!(!text.contains("dropped"), "{text}");
    }

    #[test]
    fn take_verdict_text_shows_dropped_samples() {
        let mut q = quality_summary(true, &[], None);
        q.dropped_samples = 128;
        let text = take_verdict_text(&q);
        assert!(text.contains("dropped 128 samples"), "{text}");
    }

    #[test]
    fn take_verdict_text_explains_warning_and_unknown_evidence() {
        use crate::quality::{
            QualityBand, QualityFinding, QualityMethod, QualityMetric, QualityPolicy,
            QualityStatus, QualityThreshold, QualityUnit, ThresholdComparison,
        };

        let mut q = quality_summary(false, &["high harmonic distortion"], None);
        q.findings = vec![
            QualityFinding {
                policy: QualityPolicy::SweepAnalysis,
                metric: QualityMetric::TotalHarmonicDistortion,
                status: QualityStatus::Warning,
                measured: Some(12.5),
                unit: QualityUnit::Percent,
                band: Some(QualityBand {
                    low_hz: 100.0,
                    high_hz: 20_000.0,
                }),
                method: QualityMethod::FarinaHarmonicImpulseResponse,
                threshold: Some(QualityThreshold {
                    comparison: ThresholdComparison::AtMost,
                    value: 10.0,
                }),
                unknown_reason: None,
                recommendation: Some("reduce playback level and repeat".into()),
                message: "harmonic distortion exceeded the limit".into(),
            },
            QualityFinding {
                policy: QualityPolicy::CoherentCapture,
                metric: QualityMetric::BroadbandSnr,
                status: QualityStatus::Unknown,
                measured: None,
                unit: QualityUnit::Decibels,
                band: None,
                method: QualityMethod::BroadbandPowerRatio,
                threshold: Some(QualityThreshold {
                    comparison: ThresholdComparison::AtLeast,
                    value: 30.0,
                }),
                unknown_reason: Some("the silent window is digital silence".into()),
                recommendation: Some("check the input noise-floor path".into()),
                message: "SNR unavailable".into(),
            },
            QualityFinding {
                policy: QualityPolicy::SweepAnalysis,
                metric: QualityMetric::Clipping,
                status: QualityStatus::Pass,
                measured: Some(0.0),
                unit: QualityUnit::Count,
                band: None,
                method: QualityMethod::FullScaleSampleCount,
                threshold: None,
                unknown_reason: None,
                recommendation: None,
                message: "no clipping detected".into(),
            },
        ];

        let text = take_verdict_text(&q);
        assert!(text.contains("THD warning"), "{text}");
        assert!(text.contains("12.50 %"), "{text}");
        assert!(text.contains("Farina harmonic IR"), "{text}");
        assert!(text.contains("threshold ≤ 10.00 %"), "{text}");
        assert!(text.contains("100–20000 Hz"), "{text}");
        assert!(text.contains("reduce playback level and repeat"), "{text}");
        assert!(text.contains("broadband SNR unknown"), "{text}");
        assert!(text.contains("threshold ≥ 30.00 dB"), "{text}");
        assert!(
            text.contains("the silent window is digital silence"),
            "{text}"
        );
        assert!(text.contains("check the input noise-floor path"), "{text}");
        assert!(!text.contains("clipping pass"), "{text}");
        assert!(
            text.contains("high harmonic distortion"),
            "legacy issue was lost: {text}"
        );
    }

    #[test]
    fn clean_take_verdict_includes_passing_metric_evidence() {
        use crate::quality::{
            QualityFinding, QualityMethod, QualityMetric, QualityPolicy, QualityStatus,
            QualityThreshold, QualityUnit, ThresholdComparison,
        };

        let mut quality = quality_summary(true, &[], None);
        quality.findings.push(QualityFinding {
            policy: QualityPolicy::SweepAnalysis,
            metric: QualityMetric::TotalHarmonicDistortion,
            status: QualityStatus::Pass,
            measured: Some(1.25),
            unit: QualityUnit::Percent,
            band: None,
            method: QualityMethod::FarinaHarmonicImpulseResponse,
            threshold: Some(QualityThreshold {
                comparison: ThresholdComparison::AtMost,
                value: 10.0,
            }),
            unknown_reason: None,
            recommendation: None,
            message: "THD below advisory threshold".into(),
        });

        let text = take_verdict_text(&quality);
        assert!(text.contains("THD pass"), "{text}");
        assert!(text.contains("1.25 %"), "{text}");
        assert!(text.contains("Farina harmonic IR"), "{text}");
        assert!(text.contains("threshold ≤ 10.00 %"), "{text}");
        assert!(text.contains("coherence 0.71"), "{text}");
        assert!(text.contains("SNR 18.2 dB"), "{text}");
    }

    #[test]
    fn take_verdict_text_distinguishes_drift_unavailable_from_zero() {
        // Task-7 review carry-forward: drift None must render as unavailable,
        // never as "0 ppm".
        let none = take_verdict_text(&quality_summary(true, &[], None));
        assert!(none.contains("drift n/a"), "{none}");
        assert!(!none.contains("0 ppm"), "{none}");

        let zero = take_verdict_text(&quality_summary(true, &[], Some(0.0)));
        assert!(zero.contains("drift +0 ppm"), "{zero}");
        assert!(!zero.contains("n/a"), "{zero}");
    }

    #[test]
    fn take_verdict_text_marks_drift_correction() {
        let mut q = quality_summary(true, &[], Some(30.0));
        q.drift_corrected = true;
        assert!(take_verdict_text(&q).contains("drift +30 ppm (corrected)"));
    }

    #[test]
    fn take_quality_cell_marks_review_accepted_and_clean() {
        use crate::recording_types::ChannelRecordingState::*;
        let warn = quality_summary(false, &["low coherence"], None);
        let clean = quality_summary(true, &[], Some(5.0));
        let mk = |q: TakeQualitySummary| RecordingResult {
            sample_rate_hz: None,
            channel: 0,
            wav_path: None,
            csv_path: None,
            frequencies: vec![],
            magnitude_db: vec![],
            phase_deg: vec![],
            impulse_response: None,
            impulse_time_ms: None,
            thd_percent: None,
            harmonic_distortion_db: None,
            excess_group_delay_ms: None,
            rt60_ms: None,
            clarity_c50_db: None,
            clarity_c80_db: None,
            spectrogram_db: None,
            quality: Some(q),
            stimulus_config: None,
        };
        let warned = mk(warn);
        let clean_res = mk(clean);
        assert_eq!(
            take_quality_cell(ReviewNeeded, Some(&warned)),
            "REVIEW 0.42"
        );
        // Accepted-with-warning stays distinct from a clean Done.
        assert_eq!(take_quality_cell(Done, Some(&warned)), "OK* 0.42");
        assert_eq!(take_quality_cell(Done, Some(&clean_res)), "OK 0.42");
        // Legacy / loaded results without quality data say so explicitly.
        assert_eq!(take_quality_cell(Done, None), "no data");
        assert_eq!(take_quality_cell(Empty, None), "");
        assert_eq!(take_quality_cell(Recording, None), "");
    }

    #[test]
    fn dropout_warning_only_when_samples_dropped() {
        assert!(dropout_warning(0).is_none());
        let w = dropout_warning(512).expect("warning for dropped samples");
        assert!(w.contains("512"), "{w}");
        assert!(w.contains("dropped during capture"), "{w}");
        assert!(w.contains("re-measuring"), "{w}");
    }

    #[test]
    fn session_quality_summary_one_line_per_recorded_channel() {
        use crate::recording_types::ChannelRecordingState;
        let mut clean = ChannelRecording::new(0, "FL".to_string());
        clean.state = ChannelRecordingState::Done;
        let mut clean_q = quality_summary(true, &[], None);
        clean_q.score = 0.95;
        clean_q.rejected_count = 0;
        clean.result = Some(RecordingResult {
            sample_rate_hz: None,
            channel: 0,
            wav_path: None,
            csv_path: None,
            frequencies: vec![],
            magnitude_db: vec![],
            phase_deg: vec![],
            impulse_response: None,
            impulse_time_ms: None,
            thd_percent: None,
            harmonic_distortion_db: None,
            excess_group_delay_ms: None,
            rt60_ms: None,
            clarity_c50_db: None,
            clarity_c80_db: None,
            spectrogram_db: None,
            quality: Some(clean_q),
            stimulus_config: None,
        });

        let mut bad = ChannelRecording::new(1, "FR (Pos 2)".to_string());
        bad.state = ChannelRecordingState::ReviewNeeded;
        let mut bad_q = quality_summary(false, &["clipping detected"], None);
        bad_q.dropped_samples = 128;
        bad.result = Some(RecordingResult {
            sample_rate_hz: None,
            channel: 1,
            wav_path: None,
            csv_path: None,
            frequencies: vec![],
            magnitude_db: vec![],
            phase_deg: vec![],
            impulse_response: None,
            impulse_time_ms: None,
            thd_percent: None,
            harmonic_distortion_db: None,
            excess_group_delay_ms: None,
            rt60_ms: None,
            clarity_c50_db: None,
            clarity_c80_db: None,
            spectrogram_db: None,
            quality: Some(bad_q),
            stimulus_config: None,
        });

        // Channels without a result yet are not listed.
        let pending = ChannelRecording::new(2, "C".to_string());

        let lines = session_quality_summary(&[clean, bad, pending]);
        assert_eq!(lines.len(), 2);
        assert!(
            lines[0].starts_with("FL: OK (4/4 sweeps) — score 0.95"),
            "{}",
            lines[0]
        );
        assert!(lines[0].contains("4/4 sweeps"), "{}", lines[0]);
        assert!(lines[1].starts_with("FR (Pos 2): REVIEW"), "{}", lines[1]);
        assert!(lines[1].contains("clipping detected"), "{}", lines[1]);
        assert!(lines[1].contains("128 dropped samples"), "{}", lines[1]);
    }

    #[test]
    fn session_quality_summary_excludes_error_channels_with_stale_results() {
        // Task-9 review B2: a channel whose retake failed (state = Error)
        // keeps its previous Done result in memory; the session summary must
        // not list that stale quality.
        use crate::recording_types::ChannelRecordingState;
        let mut stale = ChannelRecording::new(0, "FL".to_string());
        stale.state = ChannelRecordingState::Error;
        stale.result = Some(RecordingResult {
            sample_rate_hz: None,
            channel: 0,
            wav_path: None,
            csv_path: None,
            frequencies: vec![],
            magnitude_db: vec![],
            phase_deg: vec![],
            impulse_response: None,
            impulse_time_ms: None,
            thd_percent: None,
            harmonic_distortion_db: None,
            excess_group_delay_ms: None,
            rt60_ms: None,
            clarity_c50_db: None,
            clarity_c80_db: None,
            spectrogram_db: None,
            quality: Some(quality_summary(true, &[], None)),
            stimulus_config: None,
        });

        assert_eq!(
            session_quality_summary(&[stale.clone()]),
            [] as [std::string::String; 0]
        );

        // The structured view (task-9 review B3) reports the same filtering
        // plus a typed kind per line.
        stale.state = ChannelRecordingState::Done;
        let lines = session_quality_lines(std::slice::from_ref(&stale));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].1, SessionQualityLineKind::Ok);
        stale.state = ChannelRecordingState::ReviewNeeded;
        let lines = session_quality_lines(std::slice::from_ref(&stale));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].1, SessionQualityLineKind::Ok);
    }

    #[test]
    fn accepted_num_sweeps_for_save_is_min_over_done_channels() {
        use crate::recording_types::ChannelRecordingState;
        let mk = |state, accepted| {
            let mut ch = ChannelRecording::new(0, "L".to_string());
            ch.state = state;
            let mut q = quality_summary(true, &[], None);
            q.accepted_count = accepted;
            ch.result = Some(RecordingResult {
                sample_rate_hz: None,
                channel: 0,
                wav_path: None,
                csv_path: None,
                frequencies: vec![],
                magnitude_db: vec![],
                phase_deg: vec![],
                impulse_response: None,
                impulse_time_ms: None,
                thd_percent: None,
                harmonic_distortion_db: None,
                excess_group_delay_ms: None,
                rt60_ms: None,
                clarity_c50_db: None,
                clarity_c80_db: None,
                spectrogram_db: None,
                quality: Some(q),
                stimulus_config: None,
            });
            ch
        };

        // No captures → None (legacy "unknown" semantics preserved).
        assert_eq!(accepted_num_sweeps_for_save(&[]), None);
        let legacy = ChannelRecording::new(0, "L".to_string());
        assert_eq!(accepted_num_sweeps_for_save(&[legacy]), None);

        // Min across Done channels; ReviewNeeded takes do not count (the
        // user has not accepted them yet).
        let channels = vec![
            mk(ChannelRecordingState::Done, 4),
            mk(ChannelRecordingState::Done, 3),
            mk(ChannelRecordingState::ReviewNeeded, 1),
        ];
        assert_eq!(accepted_num_sweeps_for_save(&channels), Some(3));
    }

    #[test]
    fn summarize_take_quality_maps_capture_analysis_fields() {
        use math_audio_dsp::analysis::{
            AnalysisResult, ClippingInfo, ClockDriftEstimate, MeasurementQualityReport,
        };
        let result = AnalysisResult {
            frequencies: vec![],
            spl_db: vec![],
            phase_deg: vec![],
            estimated_lag_samples: 0,
            impulse_response: vec![],
            impulse_time_ms: vec![],
            excess_group_delay_ms: vec![],
            thd_percent: vec![],
            harmonic_distortion_db: vec![],
            rt60_ms: vec![],
            clarity_c50_db: vec![],
            clarity_c80_db: vec![],
            spectrogram_db: vec![],
        };
        let report = MeasurementQualityReport {
            trustworthy: false,
            score: 0.3,
            quality_data_complete: true,
            missing_metrics: vec![],
            lag_confidence: 0.9,
            mean_coherence: Some(0.66),
            snr_db: vec![10.0, 20.0],
            median_snr_db: Some(15.0),
            clipping: ClippingInfo {
                clipped_samples: 3,
                non_finite_samples: 0,
                total_samples: 1000,
                fraction: 0.003,
            },
            issues: vec!["clipping detected".to_string()],
        };
        let capture = CaptureAnalysis {
            result,
            quality: report,
            quality_findings: Vec::new(),
            drift: Some(ClockDriftEstimate {
                ppm: -45.0,
                start_lag_samples: 10,
                end_lag_samples: 20,
                confidence: 0.8,
            }),
            drift_corrected: true,
            dropped_samples: 7,
            accepted_count: 3,
            rejected_count: 1,
        };

        let q = summarize_take_quality(&capture);
        assert!(!q.trustworthy);
        assert_eq!(q.score, 0.3);
        assert_eq!(q.issues, vec!["clipping detected".to_string()]);
        assert_eq!(q.mean_coherence, Some(0.66));
        assert_eq!(q.median_snr_db, Some(15.0));
        assert_eq!(q.clip_fraction, 0.003);
        assert_eq!(q.drift_ppm, Some(-45.0));
        assert!(q.drift_corrected);
        assert_eq!(q.dropped_samples, 7);
        assert_eq!(q.accepted_count, 3);
        assert_eq!(q.rejected_count, 1);

        let mut capture_with_unknown = capture;
        capture_with_unknown.quality.trustworthy = true;
        capture_with_unknown.quality_findings = vec![crate::quality::QualityFinding::unknown(
            crate::quality::QualityPolicy::SingleTake,
            crate::quality::QualityMetric::BroadbandSnr,
            crate::quality::QualityUnit::Decibels,
            crate::quality::QualityMethod::BroadbandPowerRatio,
            "the pre-silence window is digital silence",
            Some("check the input noise-floor path".into()),
            "SNR could not be measured",
        )];
        let q = summarize_take_quality(&capture_with_unknown);
        assert!(!q.trustworthy, "unknown required evidence must need review");
        assert_eq!(q.findings, capture_with_unknown.quality_findings);

        // Drift unavailable stays None — never collapses to 0 ppm.
        let capture_no_drift = CaptureAnalysis {
            drift: None,
            ..capture_with_unknown
        };
        assert_eq!(summarize_take_quality(&capture_no_drift).drift_ppm, None);
    }

    #[test]
    fn position_guidance_anchors_first_position_at_mlp() {
        let first = position_guidance(0, 3);
        assert!(first.contains("Position 1 of 3"), "{first}");
        assert!(first.contains("main listening position"), "{first}");
        let later = position_guidance(1, 3);
        assert!(later.contains("position 2 of 3"), "{later}");
        assert!(later.contains("60 cm"), "{later}");
    }
}
