//! Worker-thread acquisition and durable raw artifacts for multi-microphone sessions.
//!
//! Raw acquisition is intentionally not an import-ready corrected measurement.
//! The journal records pending clock/calibration processing rather than inventing
//! timing certainty. Completed sources survive a later device failure or cancel.

use super::protocol::{CaptureStimulusLayout, prepare_capture_stimulus};
use super::{CaptureSessionPlan, ValidatedCaptureSession};
use crate::recording_helpers::save_recording_session_json;
use crate::signal_recorder::multi_capture::{
    CaptureInput, CaptureOutputSegment, MultiCaptureRequest, MultiCaptureResult,
    capture_multidevice,
};
use crate::signal_recorder::{CancelFlag, write_wav_file};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read as _;
use std::path::Path;
use std::sync::atomic::Ordering;

/// Raw acquisition state; completion does not imply coherent measurement validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawCaptureStatus {
    /// One or more sources remain to be recorded.
    Capturing,
    /// All raw takes exist; timing and calibrated analysis remain pending.
    RawComplete,
    /// User cancellation stopped acquisition.
    Cancelled,
    /// Acquisition or artifact writing failed.
    Failed,
}

/// An immutable calibration snapshot associated with one physical microphone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureCalibration {
    /// Microphone identity from the session declaration.
    pub microphone_id: String,
    /// Snapshot file relative to the capture directory.
    pub file: String,
    /// SHA-256 of the exact snapshot bytes.
    pub sha256: String,
}

/// One source/microphone pair captured on its original device clock.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawCaptureTake {
    /// Stable identity within the immutable raw journal.
    #[serde(default)]
    pub take_id: String,
    /// Zero-based acquisition repeat declared by the session plan.
    #[serde(default)]
    pub repeat_index: u32,
    /// Actual device ID returned by the audio backend.
    pub device_id: String,
    /// Actual output device ID returned by the audio backend.
    pub output_device_id: String,
    /// Source label from the session declaration.
    pub source_id: String,
    /// Microphone label from the session declaration.
    pub microphone_id: String,
    /// Raw float WAV relative to the session directory.
    pub wav_file: String,
    /// Number of raw device-clock samples, including pre/post-roll.
    pub samples: usize,
    /// Negotiated input PCM format before float conversion.
    pub input_sample_format: String,
    /// Negotiated playback PCM format.
    pub output_sample_format: String,
    /// Peak absolute raw amplitude; values at or above 1 require clipping review.
    pub peak_amplitude: f32,
    /// Number of raw samples at or above full scale.
    pub clipped_samples: usize,
}

/// Completed delay-probe pass journaled with the sweeps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureProbePass {
    /// Mono probe WAV relative to the session directory.
    pub wav_file: String,
    /// Recording microphone identity (always the first plan microphone).
    pub microphone_id: String,
    /// Per-channel arrivals and alignment delays.
    pub results: crate::signal_recorder::ProbeDelayResults,
}

/// Completed bass-anchor pass journaled with the sweeps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureBassAnchorPass {
    /// Anchor WAV relative to the session directory (mic + loopback
    /// stereo when a loopback channel was recorded, else mono).
    pub wav_file: String,
    /// Recording microphone identity (always the first plan microphone).
    pub microphone_id: String,
    /// Loopback input channel on the microphone's device, when recorded.
    pub loopback_channel: Option<u16>,
    /// Per-channel low-frequency phase evidence.
    pub results: crate::signal_recorder::BassAnchorResults,
}

/// One recorded loopback take, backing CTC raw-sweep alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureLoopbackTake {
    /// Zero-based repeat index within the session.
    pub repeat_index: u32,
    /// Source identity playing during the take.
    pub source_id: String,
    /// Mono loopback WAV relative to the session directory.
    pub wav_file: String,
    /// Recorded sample count.
    pub samples: usize,
    /// Negotiated input device identity.
    pub device_id: String,
    /// Negotiated input sample format.
    pub input_sample_format: String,
    /// Peak absolute amplitude in linear units.
    pub peak_amplitude: f32,
    /// Samples at or above digital full scale.
    pub clipped_samples: usize,
}

/// Durable acquisition journal, separate from corrected recording imports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawCaptureManifest {
    /// Raw artifact schema version.
    pub version: u32,
    /// Frozen user declaration, including geometry, gains, and orientations.
    pub plan: CaptureSessionPlan,
    /// Loopback takes, one per repeat/source when the plan declares CTC.
    #[serde(default)]
    pub loopback_files: Vec<CaptureLoopbackTake>,
    /// Completed probe pass; absent when the plan declares none.
    #[serde(default)]
    pub probe_pass: Option<CaptureProbePass>,
    /// Completed anchor pass; absent when the plan declares none.
    #[serde(default)]
    pub bass_anchor_pass: Option<CaptureBassAnchorPass>,
    /// Explicit timing-chirp and sweep positions on the stimulus clock.
    pub stimulus: CaptureStimulusLayout,
    /// Fixed chirp output route; absent in older journals that used each source.
    #[serde(default)]
    pub timing_reference_output_channel: Option<u16>,
    /// Raw acquisition lifecycle state.
    pub status: RawCaptureStatus,
    /// Processing still required before measurement import/coherent consumers.
    pub pending_processing: Vec<String>,
    /// Calibration snapshots made before opening devices.
    pub calibrations: Vec<CaptureCalibration>,
    /// Complete raw source/microphone recordings saved so far.
    pub takes: Vec<RawCaptureTake>,
    /// Failure or cancellation reason, when acquisition did not complete.
    pub error: Option<String>,
}

/// Progress notification emitted on the caller's worker thread.
#[derive(Debug, Clone)]
pub struct CaptureProgress {
    /// Zero-based index of the source about to be captured.
    pub source_index: usize,
    /// Total number of sequential sources.
    pub source_count: usize,
    /// Source label for frontend display.
    pub source_id: String,
    /// Zero-based independent repeat currently being recorded.
    pub repeat_index: u32,
    /// Total repeats declared by the session plan.
    pub repeat_count: u32,
}

pub(crate) fn stable_take_id(source_id: &str, microphone_id: &str, repeat_index: u32) -> String {
    let mut identity = Sha256::new();
    identity.update(repeat_index.to_be_bytes());
    for label in [source_id, microphone_id] {
        identity.update((label.len() as u64).to_be_bytes());
        identity.update(label.as_bytes());
    }
    format!(
        "take-r{repeat_index:03}-{}",
        super::digest_lower_hex(identity.finalize().as_ref())
    )
}

fn save_manifest(directory: &Path, manifest: &RawCaptureManifest) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(manifest)
        .map_err(|e| format!("cannot serialize capture journal: {e}"))?;
    save_recording_session_json(&directory.join("capture-raw.json"), &json)
        .map_err(|e| format!("cannot save capture journal: {e}"))
}

/// Record all declared sources with simultaneous microphone input streams.
///
/// Resolve calibration paths against `plan_directory`. `output_directory` must
/// not already exist; existing recordings are never overwritten. Run on a worker
/// thread. Cancellation retains complete sources with an explicit cancelled state.
///
/// Probe every plan microphone awake before the take loop.
///
/// One wake request exercises all input streams plus the output stream on the
/// first source channel; every mic must lag-lock the burst before the session
/// proceeds. Retries inside `budget`, then proceeds anyway: a sleeping chain
/// usually answers within one or two bursts, while a truly dead chain fails
/// fast on the first real take with the backend error. Returns `Err` only on
/// cancellation.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "The positive fixed wake tail is rounded up to a whole number of audio frames."
)]
fn wake_session_chain(
    capture: &mut impl FnMut(MultiCaptureRequest, &CancelFlag) -> Result<MultiCaptureResult, String>,
    session: &ValidatedCaptureSession,
    cancel: &CancelFlag,
    budget: std::time::Duration,
) -> Result<(), String> {
    use crate::signal_recorder::wake::{WAKE_ERROR_BACKOFF_MS, WAKE_TAIL_SECS, wake_burst_samples};
    use std::thread::sleep;
    use std::time::Instant;

    let rate = session.plan().sample_rate_hz;
    let burst = wake_burst_samples(rate);
    let Some(first_source) = session.plan().sources.first() else {
        return Ok(());
    };
    let inputs: Vec<CaptureInput> = session
        .plan()
        .microphones
        .iter()
        .map(|mic| CaptureInput {
            device: mic.device.clone(),
            channel: mic.input_channel,
        })
        .collect();
    if burst.is_empty() || inputs.is_empty() {
        return Ok(());
    }
    let mut stimulus = burst.clone();
    stimulus.extend(std::iter::repeat_n(
        0.0,
        (WAKE_TAIL_SECS * f64::from(rate)).ceil() as usize,
    ));
    let minimum =
        math_audio_dsp::analysis::MeasurementQualityConfig::default().minimum_lag_confidence;
    let deadline = Instant::now() + budget;
    let mut attempts = 0_u32;
    while Instant::now() < deadline {
        attempts += 1;
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let response = capture(
            MultiCaptureRequest {
                inputs: inputs.clone(),
                output_device: session.plan().output_device.clone(),
                output_channel: first_source.output_channel,
                output_overrides: Vec::new(),
                sample_rate_hz: rate,
                stimulus: stimulus.clone(),
            },
            cancel,
        );
        match response {
            Ok(captured) if captured.recordings.len() == inputs.len() => {
                let awake = captured.recordings.iter().all(|recording| {
                    math_audio_dsp::analysis::estimate_lag_with_confidence(&burst, recording)
                        .is_ok_and(|lag| lag.confidence >= minimum)
                });
                if awake {
                    if attempts > 1 {
                        log::info!("Audio chain awake after {attempts} wake-up attempts");
                    } else {
                        log::debug!("Audio chain awake on first wake-up attempt");
                    }
                    return Ok(());
                }
            }
            Err(_) => sleep(std::time::Duration::from_millis(WAKE_ERROR_BACKOFF_MS)),
            Ok(_) => {}
        }
    }
    log::warn!(
        "Audio chain did not respond within {:.0}s of wake-up probes — proceeding; the first take will fail fast on a dead chain",
        budget.as_secs_f64(),
    );
    Ok(())
}

/// Record all declared sources with simultaneous microphone input streams.
///
/// Resolve calibration paths against `plan_directory`. `output_directory` must
/// not already exist; existing recordings are never overwritten. Run on a worker
/// thread. Cancellation retains complete sources with an explicit cancelled state.
///
/// # Errors
/// Returns configuration, calibration, device, cancellation, or filesystem errors.
/// An error after acquisition starts preserves its journal and completed sources.
pub fn record_capture_session(
    session: &ValidatedCaptureSession,
    plan_directory: &Path,
    output_directory: &Path,
    cancel: &CancelFlag,
    progress: impl FnMut(CaptureProgress),
) -> Result<RawCaptureManifest, String> {
    let manifest = record_with(
        session,
        plan_directory,
        output_directory,
        cancel,
        progress,
        capture_multidevice,
    )?;
    run_auxiliary_passes(
        session,
        output_directory,
        cancel,
        manifest,
        crate::signal_recorder::probe_channel_delays_with_recording,
        crate::signal_recorder::run_bass_anchor_with_recording,
    )
}

/// Probe WAV name inside raw and processed session directories.
const PROBE_WAV_FILE: &str = "probe.wav";
/// Bass-anchor WAV name inside raw and processed session directories.
const BASS_ANCHOR_WAV_FILE: &str = "bass-anchor.wav";

/// Run declared probe/anchor passes after the sweeps and journal them.
///
/// Channels follow the plan sources in order; the first microphone records.
/// A declared pass that fails marks the session Failed with its reason while
/// preserving completed sweeps for explicit-selection processing. Runners are
/// injected so hardware-free tests can supply canned results.
pub(crate) fn run_auxiliary_passes(
    session: &ValidatedCaptureSession,
    output_directory: &Path,
    cancel: &CancelFlag,
    mut manifest: RawCaptureManifest,
    mut probe_runner: impl FnMut(
        &[u16],
        &[String],
        u32,
        f32,
        f32,
        Option<&str>,
        Option<&str>,
        u16,
        &Path,
        f32,
        Option<CancelFlag>,
    ) -> Result<crate::signal_recorder::ProbeDelayResults, String>,
    mut anchor_runner: impl FnMut(
        &[u16],
        &[String],
        u32,
        f32,
        f32,
        f32,
        u16,
        f32,
        Option<&str>,
        Option<&str>,
        u16,
        Option<u16>,
        &Path,
        f32,
        Option<CancelFlag>,
    ) -> Result<crate::signal_recorder::BassAnchorResults, String>,
) -> Result<RawCaptureManifest, String> {
    let plan = session.plan();
    if plan.probe.is_none() && plan.bass_anchor.is_none() {
        return Ok(manifest);
    }
    let fail = |manifest: &mut RawCaptureManifest, error: String| -> String {
        manifest.status = if error == "cancelled" {
            RawCaptureStatus::Cancelled
        } else {
            RawCaptureStatus::Failed
        };
        manifest.error = Some(error.clone());
        save_manifest(output_directory, manifest).map_or_else(
            |save_error| format!("{error}; additionally {save_error}"),
            |()| error.clone(),
        )
    };
    let channels: Vec<u16> = plan
        .sources
        .iter()
        .map(|source| source.output_channel)
        .collect();
    let names: Vec<String> = plan
        .sources
        .iter()
        .map(|source| source.id.clone())
        .collect();
    let microphone = &plan.microphones[0];
    if cancel.load(Ordering::Relaxed) {
        let error = fail(&mut manifest, "cancelled".into());
        return Err(error);
    }
    if let Some(probe) = &plan.probe {
        let wav = output_directory.join(PROBE_WAV_FILE);
        match probe_runner(
            &channels,
            &names,
            plan.sample_rate_hz,
            probe.probe_duration_ms,
            probe.silence_duration_ms,
            Some(plan.output_device.as_str()),
            Some(microphone.device.as_str()),
            microphone.input_channel,
            &wav,
            probe.signal_level_db,
            Some(CancelFlag::clone(cancel)),
        ) {
            Ok(results) => {
                manifest.probe_pass = Some(CaptureProbePass {
                    wav_file: PROBE_WAV_FILE.into(),
                    microphone_id: microphone.id.clone(),
                    results,
                });
                save_manifest(output_directory, &manifest)?;
            }
            Err(error) => return Err(fail(&mut manifest, error)),
        }
    }
    if cancel.load(Ordering::Relaxed) {
        let error = fail(&mut manifest, "cancelled".into());
        return Err(error);
    }
    if let Some(anchor) = &plan.bass_anchor {
        let loopback_channel = plan.loopback.as_ref().map(|route| route.input_channel);
        let wav = output_directory.join(BASS_ANCHOR_WAV_FILE);
        match anchor_runner(
            &channels,
            &names,
            plan.sample_rate_hz,
            anchor.bass_freq_hz,
            anchor.bass_duration_s,
            anchor.fade_ms,
            anchor.num_windows,
            anchor.silence_duration_ms,
            Some(plan.output_device.as_str()),
            Some(microphone.device.as_str()),
            microphone.input_channel,
            loopback_channel,
            &wav,
            anchor.signal_level_db,
            Some(CancelFlag::clone(cancel)),
        ) {
            Ok(results) => {
                manifest.bass_anchor_pass = Some(CaptureBassAnchorPass {
                    wav_file: BASS_ANCHOR_WAV_FILE.into(),
                    microphone_id: microphone.id.clone(),
                    loopback_channel,
                    results,
                });
                save_manifest(output_directory, &manifest)?;
            }
            Err(error) => return Err(fail(&mut manifest, error)),
        }
    }
    Ok(manifest)
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep the ordered acquisition and artifact-processing phases together so cleanup and provenance remain explicit."
)]
fn record_with(
    session: &ValidatedCaptureSession,
    plan_directory: &Path,
    output_directory: &Path,
    cancel: &CancelFlag,
    mut progress: impl FnMut(CaptureProgress),
    mut capture: impl FnMut(MultiCaptureRequest, &CancelFlag) -> Result<MultiCaptureResult, String>,
) -> Result<RawCaptureManifest, String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    let stimulus = prepare_capture_stimulus(session)?;
    // Parse copies of the exact bytes we hash/save, not a second read of mutable
    // user calibration files. Every calibration must load before audio starts.
    let snapshots =
        tempfile::tempdir().map_err(|e| format!("cannot stage calibration files: {e}"))?;
    let mut calibrations = Vec::new();
    for (index, mic) in session.plan().microphones.iter().enumerate() {
        let path = plan_directory.join(&mic.calibration_file);
        // Calibration is a small text curve. Bound external file reads to 1 MiB.
        let file = std::fs::File::open(&path)
            .map_err(|e| format!("cannot read calibration for {}: {e}", mic.id))?;
        let mut bytes = Vec::new();
        file.take(1_048_577)
            .read_to_end(&mut bytes)
            .map_err(|e| format!("cannot read calibration for {}: {e}", mic.id))?;
        if bytes.len() > 1_048_576 {
            return Err(format!("calibration for {} exceeds 1 MiB", mic.id));
        }
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("txt");
        if !matches!(extension, "txt" | "csv" | "TXT" | "CSV") {
            return Err(format!(
                "calibration for {} must be a .txt or .csv curve",
                mic.id
            ));
        }
        let file = format!("calibration-{index:02}.{}", extension.to_ascii_lowercase());
        let snapshot = snapshots.path().join(&file);
        std::fs::write(&snapshot, &bytes).map_err(|e| e.to_string())?;
        math_audio_dsp::analysis::MicrophoneCompensation::from_file(&snapshot)
            .map_err(|e| format!("invalid calibration for {}: {e}", mic.id))?;
        calibrations.push(CaptureCalibration {
            microphone_id: mic.id.clone(),
            file,
            sha256: super::calibration_sha256(&bytes),
        });
    }
    std::fs::create_dir(output_directory)
        .map_err(|e| format!("cannot create new capture directory: {e}"))?;
    let mut manifest = RawCaptureManifest {
        version: 1,
        plan: session.plan().clone(),
        stimulus: stimulus.layout,
        timing_reference_output_channel: Some(super::timing_emitter_channel(session)),
        status: RawCaptureStatus::Capturing,
        pending_processing: vec![
            "clock_correction".into(),
            "calibrated_analysis".into(),
            "take_quality".into(),
        ],
        calibrations,
        takes: Vec::new(),
        loopback_files: Vec::new(),
        error: None,
        probe_pass: None,
        bass_anchor_pass: None,
    };
    save_manifest(output_directory, &manifest)?;
    let result = (|| -> Result<(), String> {
        for calibration in &manifest.calibrations {
            std::fs::copy(
                snapshots.path().join(&calibration.file),
                output_directory.join(&calibration.file),
            )
            .map_err(|e| format!("cannot save calibration snapshot: {e}"))?;
        }
        write_wav_file(
            &output_directory.join("stimulus.wav"),
            &stimulus.samples,
            session.plan().sample_rate_hz,
            1,
        )?;
        write_wav_file(
            &output_directory.join("timing-chirp.wav"),
            &stimulus.timing_chirp,
            session.plan().sample_rate_hz,
            1,
        )?;
        wake_session_chain(
            &mut capture,
            session,
            cancel,
            std::time::Duration::from_secs_f64(crate::signal_recorder::wake::WAKE_BUDGET_SECS),
        )?;
        for repeat_index in 0..session.plan().repeat_count {
            for (source_index, source) in session.plan().sources.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    return Err("cancelled".into());
                }
                progress(CaptureProgress {
                    source_index,
                    source_count: session.plan().sources.len(),
                    source_id: source.id.clone(),
                    repeat_index,
                    repeat_count: session.plan().repeat_count,
                });
                let mut inputs: Vec<CaptureInput> = session
                    .plan()
                    .microphones
                    .iter()
                    .map(|mic| CaptureInput {
                        device: mic.device.clone(),
                        channel: mic.input_channel,
                    })
                    .collect();
                // The loopback rides the same capture call so it shares the
                // take timeline; CTC assembly aligns RoomEQ to it later.
                let loopback_index = if session.plan().ctc.is_some() {
                    let route = session
                        .plan()
                        .loopback
                        .as_ref()
                        .ok_or("ctc requires a plan loopback route")?;
                    inputs.push(CaptureInput {
                        device: route.device.clone(),
                        channel: route.input_channel,
                    });
                    Some(inputs.len() - 1)
                } else {
                    None
                };
                let captured = capture(
                    MultiCaptureRequest {
                        inputs,
                        output_device: session.plan().output_device.clone(),
                        output_channel: source.output_channel,
                        output_overrides: [
                            manifest.stimulus.start_chirp_offset,
                            manifest.stimulus.end_chirp_offset,
                        ]
                        .into_iter()
                        .map(|start_frame| CaptureOutputSegment {
                            start_frame,
                            end_frame: start_frame + manifest.stimulus.chirp_samples,
                            channel: manifest
                                .timing_reference_output_channel
                                .unwrap_or(source.output_channel),
                        })
                        .collect(),
                        sample_rate_hz: session.plan().sample_rate_hz,
                        stimulus: stimulus.samples.clone(),
                    },
                    cancel,
                )?;
                let expected_inputs =
                    session.plan().microphones.len() + usize::from(loopback_index.is_some());
                if captured.sample_rate_hz != session.plan().sample_rate_hz
                    || captured.recordings.len() != expected_inputs
                    || captured.input_sample_formats.len() != expected_inputs
                    || captured.input_device_ids.len() != expected_inputs
                    || captured.input_device_ids.iter().any(String::is_empty)
                    || captured.output_device_id.is_empty()
                    || captured.recordings.iter().any(|samples| {
                        samples.len() < stimulus.samples.len()
                            || samples.iter().any(|s| !s.is_finite())
                    })
                {
                    return Err(
                        "capture backend returned mismatched rate, channels, or invalid samples"
                            .into(),
                    );
                }
                for (mic_index, (mic, samples)) in session
                    .plan()
                    .microphones
                    .iter()
                    .zip(&captured.recordings)
                    .enumerate()
                {
                    // Numeric names avoid collisions and path traversal in user labels.
                    let wav_file = format!(
                        "repeat-{repeat_index:02}-source-{source_index:02}-mic-{mic_index:02}-raw.wav"
                    );
                    write_wav_file(
                        &output_directory.join(&wav_file),
                        samples,
                        captured.sample_rate_hz,
                        1,
                    )?;
                    manifest.takes.push(RawCaptureTake {
                        take_id: stable_take_id(&source.id, &mic.id, repeat_index),
                        repeat_index,
                        device_id: captured.input_device_ids[mic_index].clone(),
                        output_device_id: captured.output_device_id.clone(),
                        source_id: source.id.clone(),
                        microphone_id: mic.id.clone(),
                        wav_file,
                        samples: samples.len(),
                        input_sample_format: captured.input_sample_formats[mic_index].clone(),
                        output_sample_format: captured.output_sample_format.clone(),
                        peak_amplitude: samples.iter().fold(0.0_f32, |peak, s| peak.max(s.abs())),
                        clipped_samples: samples.iter().filter(|s| s.abs() >= 1.0).count(),
                    });
                }
                if let Some(loopback) = loopback_index {
                    let samples = &captured.recordings[loopback];
                    let wav_file = format!(
                        "repeat-{repeat_index:02}-source-{source_index:02}-loopback-raw.wav"
                    );
                    write_wav_file(
                        &output_directory.join(&wav_file),
                        samples,
                        captured.sample_rate_hz,
                        1,
                    )?;
                    manifest.loopback_files.push(CaptureLoopbackTake {
                        repeat_index,
                        source_id: source.id.clone(),
                        wav_file,
                        samples: samples.len(),
                        device_id: captured.input_device_ids[loopback].clone(),
                        input_sample_format: captured.input_sample_formats[loopback].clone(),
                        peak_amplitude: samples.iter().fold(0.0_f32, |peak, s| peak.max(s.abs())),
                        clipped_samples: samples.iter().filter(|s| s.abs() >= 1.0).count(),
                    });
                }
                save_manifest(output_directory, &manifest)?;
            }
        }
        Ok(())
    })();
    match result {
        Ok(()) => {
            manifest.status = RawCaptureStatus::RawComplete;
            save_manifest(output_directory, &manifest)?;
            Ok(manifest)
        }
        Err(error) => {
            manifest.status = if error == "cancelled" {
                RawCaptureStatus::Cancelled
            } else {
                RawCaptureStatus::Failed
            };
            manifest.error = Some(error.clone());
            save_manifest(output_directory, &manifest)
                .map_err(|save_error| format!("{error}; additionally {save_error}"))?;
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests;
