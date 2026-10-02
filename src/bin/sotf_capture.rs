//! Standalone capture CLI over the sotf-capture library.
//!
//! Lists audio devices, captures swept-sine (or other stimulus) takes
//! through the cpal-native playback backend, and records multi-source
//! capture-session plans. Every command mirrors the corresponding
//! `sotf` frontend flow so behavior stays identical across shells.

// Rust guideline compliant 2026-02-21

use clap::{Parser, Subcommand};
use serde::{Deserialize, Deserializer};
#[cfg(test)]
use sha2::Digest;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "sotf-capture",
    version,
    about = "Acoustic measurement capture without the DAW engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
#[expect(
    clippy::large_enum_variant,
    reason = "Clap parses this command once; inline fields keep its argument definitions together"
)]
enum Command {
    /// List input and output audio devices.
    Devices {
        /// Emit the device map as JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Emit raw live level/RTA JSON frames from an explicitly selected input.
    Live {
        /// Input device ID or unique name; no default-device fallback.
        #[arg(long)]
        input_device: String,
        /// Zero-based hardware input channel.
        #[arg(long, default_value_t = 0)]
        input_channel: u16,
        /// Required hardware sample rate, without resampling.
        #[arg(long, default_value_t = 48000)]
        sample_rate: u32,
        /// Power-of-two nonoverlapping FFT block size.
        #[arg(long, default_value_t = 4096)]
        fft_size: usize,
        /// Maximum monitoring time in seconds; Ctrl-C stops early.
        #[arg(long, default_value_t = 30.0)]
        duration: f64,
        /// Calibration JSON schema 1: binding saved when calibrated, response_curve {path, format,
        /// convention, reference_frequency_hz, sha256?}, and/or rms_anchor {response_sha256,
        /// reference RMS, dB SPL, frequency, band, z weighting}. Anchor response_sha256 is required:
        /// use the saved curve digest for an anchor made with a response curve, or null only for a
        /// response-free anchor.
        /// Curve paths are relative to this JSON. Known local source: ../sotf-capture/data_tests/microphones;
        /// select a file and declare its format/sign here and orientation in machine settings.
        #[arg(long, requires = "machine_settings", value_name = "JSON")]
        calibration_profile: Option<PathBuf>,
        /// Machine JSON schema 1: input_selector, runtime_binding {microphone_id, host_api, input_device_id,
        /// channel, rate, sample format, declared gain, gain_attested, orientation}, and optional spl_band_hz.
        /// runtime_binding must exactly match the profile's saved binding; changing a route requires
        /// a new calibration.
        #[arg(long, requires = "calibration_profile", value_name = "JSON")]
        machine_settings: Option<PathBuf>,
        /// Validate the profile/settings files and print JSON without opening an audio device.
        #[arg(long, requires_all = ["calibration_profile", "machine_settings"])]
        validate_calibration: bool,
    },
    /// Capture stimulus takes for output/input channel pairs.
    Capture {
        /// Stimulus: sweep, tone, two-tone, white-noise, pink-noise,
        /// m-noise, mls, dirac.
        #[arg(long, default_value = "sweep")]
        signal: String,
        /// Stimulus duration in seconds.
        #[arg(long, default_value_t = 5.0)]
        duration: f32,
        /// Sample rate in Hz.
        #[arg(long, default_value_t = 48000)]
        sample_rate: u32,
        /// Signal channels; must be 1 (mono generation).
        #[arg(long, default_value_t = 1)]
        channels: u16,
        /// Hardware output channels, e.g. "0,1".
        #[arg(long)]
        hwaudio_send_to: String,
        /// Hardware input channels, e.g. "0" (one per send, or one shared).
        #[arg(long)]
        hwaudio_record_from: String,
        /// Output filename prefix.
        #[arg(long)]
        name: Option<String>,
        /// Output directory (created when missing).
        #[arg(long)]
        output_dir: Option<PathBuf>,
        /// Shared input/output device selector (legacy convenience).
        #[arg(long, conflicts_with_all = ["input_device", "output_device"])]
        device: Option<String>,
        /// Independent input device ID or unique name, e.g. a USB microphone.
        #[arg(long)]
        input_device: Option<String>,
        /// Independent output device ID or unique name, e.g. an audio interface.
        #[arg(long)]
        output_device: Option<String>,
        /// Tone frequency in Hz.
        #[arg(long)]
        freq: Option<f32>,
        /// Two-tone first frequency in Hz.
        #[arg(long)]
        freq1: Option<f32>,
        /// Two-tone second frequency in Hz.
        #[arg(long)]
        freq2: Option<f32>,
        /// Sweep start frequency in Hz.
        #[arg(long, default_value_t = 20.0)]
        start_freq: f32,
        /// Sweep end frequency in Hz.
        #[arg(long, default_value_t = 20000.0)]
        end_freq: f32,
        /// Stimulus amplitude in (0, 1].
        #[arg(long)]
        amp: Option<f32>,
        /// Two-tone first amplitude in (0, 1].
        #[arg(long)]
        amp1: Option<f32>,
        /// Two-tone second amplitude in (0, 1].
        #[arg(long)]
        amp2: Option<f32>,
        /// MLS order.
        #[arg(long)]
        mls_order: Option<u8>,
        /// Microphone compensation file (post-compensation; sweeps also
        /// get playback pre-compensation).
        #[arg(long)]
        microphone_compensation: Option<String>,
        /// Per-channel calibration override as CH=PATH (repeatable).
        #[arg(long = "mic-calibration")]
        mic_calibrations: Vec<String>,
        /// Emit one JSON summary line per captured pair.
        #[arg(long)]
        json: bool,
    },
    /// Validate a capture-session plan without recording.
    PlanValidate {
        /// Path to the session plan JSON.
        #[arg(long)]
        plan: PathBuf,
    },
    /// Record a capture-session plan (all microphones, all sources).
    PlanRecord {
        /// Path to the session plan JSON.
        #[arg(long)]
        plan: PathBuf,
        /// Output directory for raw takes.
        #[arg(long)]
        output: PathBuf,
    },
    /// Clock-process a saved raw session without opening audio devices.
    PlanProcess {
        /// Path to the immutable capture-raw.json journal.
        #[arg(long)]
        raw: PathBuf,
        /// New output directory for processed evidence and optional RoomEQ projection.
        #[arg(long)]
        output: PathBuf,
        /// Explicit take ID to project; repeat once per source/microphone pair.
        #[arg(long = "select-take")]
        selected_take_ids: Vec<String>,
    },
}

fn validated_amp(amp: Option<f32>, flag: &str) -> Result<f32, String> {
    let amp = amp.unwrap_or(0.5);
    if !amp.is_finite() || amp <= 0.0 || amp > 1.0 {
        return Err(format!(
            "{flag} must be in the range (0.0, 1.0] — level must be ≤ 0 dBFS to avoid clipping the stimulus, got {amp}"
        ));
    }
    Ok(amp)
}

fn list_audio_devices(json: bool) -> Result<(), String> {
    use sotf_capture::devices::get_audio_devices;
    use sotf_capture::signal_recorder::actionable_capture_error;

    let devices = get_audio_devices()
        .map_err(|e| actionable_capture_error("Failed to enumerate audio devices", &e))?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&devices).unwrap_or_default()
        );
        return Ok(());
    }
    println!("{}", "=".repeat(80));
    println!("Available Audio Devices");
    println!("{}", "=".repeat(80));
    for (title, key) in [("INPUT DEVICES:", "input"), ("OUTPUT DEVICES:", "output")] {
        println!("\n{title}");
        println!("{}", "-".repeat(80));
        if let Some(listed) = devices.get(key) {
            for (index, device) in listed.iter().enumerate() {
                let default_marker = if device.is_default { " (Default)" } else { "" };
                if let Some(config) = &device.default_config {
                    println!(
                        "  [{}] {}{} - {} ch, {} Hz, {}",
                        index,
                        device.name,
                        default_marker,
                        config.channels,
                        config.sample_rate,
                        config.sample_format
                    );
                } else {
                    println!("  [{}] {}{}", index, device.name, default_marker);
                }
                if let Some(id) = &device.device_id {
                    println!("      Device ID: {id}");
                }
            }
        }
    }
    println!("\n{}", "=".repeat(80));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_signal(
    signal: String,
    duration: f32,
    sample_rate: u32,
    channels: u16,
    hwaudio_send_to: String,
    hwaudio_record_from: String,
    name: Option<String>,
    output_dir: Option<PathBuf>,
    device: Option<String>,
    input_device: Option<String>,
    output_device: Option<String>,
    freq: Option<f32>,
    freq1: Option<f32>,
    freq2: Option<f32>,
    start_freq: f32,
    end_freq: f32,
    amp: Option<f32>,
    amp1: Option<f32>,
    amp2: Option<f32>,
    mls_order: Option<u8>,
    microphone_compensation: Option<String>,
    mic_calibration_map: HashMap<usize, String>,
    json: bool,
) -> Result<(), String> {
    use sotf_capture::recording_helpers::take_verdict_text;
    use sotf_capture::recording_helpers::{capture_signal_params, summarize_take_quality};
    use sotf_capture::signal_recorder::{
        CpalPlayback, DEFAULT_MLS_ORDER, SignalParams, SignalType,
        generate_output_filenames_stereo, generate_signal, parse_channel_list,
        prepare_measurement_signal, prepare_signal, record_and_analyze_with,
        validate_signal_params, write_temp_wav,
    };
    use std::str::FromStr;

    let output_dir = match output_dir {
        Some(dir) => dir,
        None => std::env::current_dir()
            .map_err(|e| format!("cannot determine current directory: {e}"))?,
    };
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("failed to create output directory: {e}"))?;
    if channels != 1 {
        return Err(format!(
            "Channels must be 1 (mono signal generation), got {channels}"
        ));
    }
    let signal_type = SignalType::from_str(&signal)?;
    let send_to_channels = parse_channel_list(&hwaudio_send_to)?;
    let record_from_channels = parse_channel_list(&hwaudio_record_from)?;
    if send_to_channels.is_empty() {
        return Err("hwaudio-send-to must specify at least 1 channel".to_string());
    }
    for selector in [&device, &input_device, &output_device]
        .into_iter()
        .flatten()
    {
        if selector.trim().is_empty() {
            return Err("device selectors must be nonempty when supplied".into());
        }
    }
    for channel in mic_calibration_map.keys() {
        if !record_from_channels
            .iter()
            .any(|input| usize::from(*input) == *channel)
        {
            return Err(format!(
                "microphone calibration channel {channel} is not a requested input"
            ));
        }
    }
    if send_to_channels.len() != record_from_channels.len() && record_from_channels.len() != 1 {
        return Err(format!(
            "Invalid channel configuration: {} send-to channels, {} record-from channels.\n\
             Must be either equal counts or a single shared record channel.",
            send_to_channels.len(),
            record_from_channels.len()
        ));
    }
    let params = match signal_type {
        SignalType::Tone => {
            let freq = freq.ok_or("--freq is required for tone signal")?;
            SignalParams::Tone {
                freq,
                amp: validated_amp(amp, "--amp")?,
            }
        }
        SignalType::TwoTone => {
            let freq1 = freq1.ok_or("--freq1 is required for two-tone signal")?;
            let freq2 = freq2.ok_or("--freq2 is required for two-tone signal")?;
            SignalParams::TwoTone {
                freq1,
                amp1: validated_amp(amp1, "--amp1")?,
                freq2,
                amp2: validated_amp(amp2, "--amp2")?,
            }
        }
        SignalType::Sweep => capture_signal_params(
            SignalType::Sweep,
            start_freq,
            end_freq,
            validated_amp(amp, "--amp")?,
            None,
            None,
            None,
        ),
        SignalType::WhiteNoise | SignalType::PinkNoise | SignalType::MNoise => {
            SignalParams::Noise {
                amp: validated_amp(amp, "--amp")?,
            }
        }
        SignalType::Mls => SignalParams::Mls {
            order: mls_order.unwrap_or(DEFAULT_MLS_ORDER),
            amp: validated_amp(amp, "--amp")?,
        },
        SignalType::Dirac => SignalParams::Dirac {
            amp: validated_amp(amp, "--amp")?,
        },
    };

    // Playback pre-compensation (sweeps only) runs between generation and
    // `prepare_signal`; every other path shares `prepare_measurement_signal`.
    let pre_compensation = match microphone_compensation {
        Some(ref path) => {
            use std::path::Path;
            let compensation =
                math_audio_dsp::analysis::MicrophoneCompensation::from_file(Path::new(path))?;
            if signal_type == SignalType::Sweep {
                Some(compensation)
            } else {
                None
            }
        }
        None => None,
    };
    let prepared_signal = match pre_compensation {
        Some(ref compensation) => {
            validate_signal_params(signal_type, &params, duration, sample_rate)?;
            let raw_signal = generate_signal(signal_type, &params, duration, sample_rate)?;
            let compensated =
                compensation.apply_to_sweep(&raw_signal, start_freq, end_freq, sample_rate, true);
            prepare_signal(compensated, sample_rate)
        }
        None => prepare_measurement_signal(signal_type, &params, duration, sample_rate)?,
    };

    for (index, &send_ch) in send_to_channels.iter().enumerate() {
        let record_ch = if record_from_channels.len() == 1 {
            record_from_channels[0]
        } else {
            record_from_channels[index]
        };
        let (wav_name, csv_name) = generate_output_filenames_stereo(
            name.as_deref(),
            signal_type,
            send_ch,
            record_ch,
            sample_rate,
        );
        let wav_path = output_dir.join(wav_name);
        let csv_path = output_dir.join(csv_name);
        let temp_wav = write_temp_wav(&prepared_signal, sample_rate, 1)?;
        let effective_mic_comp = mic_calibration_map
            .get(&(record_ch as usize))
            .map(String::as_str)
            .or(microphone_compensation.as_deref());
        let mut playback = CpalPlayback::new();
        let capture = record_and_analyze_with(
            &mut playback,
            temp_wav.path(),
            &wav_path,
            &prepared_signal,
            sample_rate,
            &csv_path,
            send_ch,
            record_ch,
            output_device.as_deref().or(device.as_deref()),
            input_device.as_deref().or(device.as_deref()),
            effective_mic_comp,
            None,
            1,
            None,
        )?;
        let quality = summarize_take_quality(&capture);
        let verdict = take_verdict_text(&quality);
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "version": 1,
                    "send_channel": send_ch,
                    "record_channel": record_ch,
                    "requested_output_device": output_device.as_deref().or(device.as_deref()),
                    "requested_input_device": input_device.as_deref().or(device.as_deref()),
                    "microphone_calibration": effective_mic_comp,
                    "wav": wav_path.display().to_string(),
                    "csv": csv_path.display().to_string(),
                    "trustworthy": quality.trustworthy,
                    "verdict": verdict,
                })
            );
        } else if quality.trustworthy {
            println!("pair hw{send_ch}->hw{record_ch}: complete — {verdict}");
        } else {
            println!("pair hw{send_ch}->hw{record_ch}: NEEDS REVIEW — {verdict}");
        }
        if index + 1 < send_to_channels.len() {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
    Ok(())
}

fn plan_validate(plan_path: &std::path::Path) -> Result<(), String> {
    use sotf_capture::capture_session::CaptureSessionPlan;

    let json = std::fs::read_to_string(plan_path)
        .map_err(|e| format!("cannot read capture session: {e}"))?;
    let plan: CaptureSessionPlan =
        serde_json::from_str(&json).map_err(|e| format!("cannot parse capture session: {e}"))?;
    let session = plan.validate().map_err(|e| e.to_string())?;
    println!(
        "Session plan valid: {:?}, {} microphones, {} sources, {} Hz.",
        session.plan().geometry,
        session.plan().microphones.len(),
        session.plan().sources.len(),
        session.plan().sample_rate_hz,
    );
    Ok(())
}

fn plan_record(plan_path: &std::path::Path, output: &std::path::Path) -> Result<(), String> {
    use sotf_capture::capture_session::CaptureSessionPlan;
    use sotf_capture::capture_session::record::record_capture_session;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let json = std::fs::read_to_string(plan_path)
        .map_err(|e| format!("cannot read capture session: {e}"))?;
    let plan: CaptureSessionPlan =
        serde_json::from_str(&json).map_err(|e| format!("cannot parse capture session: {e}"))?;
    let session = plan.validate().map_err(|e| e.to_string())?;
    let cancel = Arc::new(AtomicBool::new(false));
    let handler_cancel = Arc::clone(&cancel);
    ctrlc::set_handler(move || handler_cancel.store(true, Ordering::Relaxed))
        .map_err(|e| format!("cannot install recording cancellation handler: {e}"))?;
    let plan_directory = plan_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let manifest = record_capture_session(&session, plan_directory, output, &cancel, |event| {
        println!(
            "Recording repeat {}/{} — source {}/{}: {} (all microphones)",
            event.repeat_index + 1,
            event.repeat_count,
            event.source_index + 1,
            event.source_count,
            event.source_id
        );
    })?;
    println!(
        "Saved {} raw takes. Clock correction, calibrated analysis and take QA remain pending.",
        manifest.takes.len()
    );
    Ok(())
}

fn plan_process(
    raw_manifest_path: &std::path::Path,
    output: &std::path::Path,
    selected_take_ids: &[String],
) -> Result<(), String> {
    use sotf_capture::capture_session::clock::io::process_capture_session_with_selection;

    if raw_manifest_path.file_name().and_then(|name| name.to_str()) != Some("capture-raw.json") {
        return Err("--raw must name a capture-raw.json journal".into());
    }
    let raw_directory = raw_manifest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let report = process_capture_session_with_selection(raw_directory, output, selected_take_ids)?;
    if let Some(configuration) = report.recording_manifest {
        println!(
            "Saved analyzed capture selection to {}/{}.",
            output.display(),
            configuration
        );
    } else {
        println!(
            "Saved diagnostics for {:?} parent; no complete selected source/microphone matrix was published.",
            report.raw_status
        );
    }
    Ok(())
}

fn parse_mic_calibrations(values: &[String]) -> Result<HashMap<usize, String>, String> {
    let mut map = HashMap::new();
    for value in values {
        let (channel, path) = value
            .split_once('=')
            .ok_or_else(|| format!("--mic-calibration expects CH=PATH, got {value:?}"))?;
        let channel: usize = channel
            .parse()
            .map_err(|_| format!("invalid --mic-calibration channel: {channel:?}"))?;
        if channel >= 64 || path.trim().is_empty() {
            return Err("--mic-calibration requires a channel below 64 and a nonempty path".into());
        }
        if map.insert(channel, path.to_string()).is_some() {
            return Err(format!("duplicate --mic-calibration for channel {channel}"));
        }
    }
    Ok(map)
}

// The document cap keeps JSON parsing bounded; the curve cap matches the immutable profile API.
const LIVE_CALIBRATION_DOCUMENT_MAX_BYTES: usize = 64 * 1024;
const LIVE_RESPONSE_CURVE_MAX_BYTES: usize = 1_048_576;
const LIVE_CALIBRATION_DOCUMENT_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveCalibrationDocument {
    schema_version: u32,
    /// Binding recorded when this calibration and optional RMS anchor were established.
    binding: sotf_capture::live_level::LiveCalibrationBinding,
    response_curve: Option<LiveResponseCurveDocument>,
    rms_anchor: Option<LiveRmsAnchorDocument>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveResponseCurveDocument {
    path: PathBuf,
    format: sotf_capture::live_level::LiveResponseCurveFormat,
    convention: sotf_capture::live_level::LiveResponseCurveConvention,
    reference_frequency_hz: f64,
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveRmsAnchorDocument {
    /// Digest of the exact response curve used for this anchor, or explicit null without a curve.
    #[serde(deserialize_with = "deserialize_required_response_sha256")]
    response_sha256: Option<String>,
    reference_rms_full_scale: f64,
    reference_level_db_spl: f64,
    reference_frequency_hz: f64,
    band_hz: [f64; 2],
    weighting: sotf_capture::live_level::LiveSplWeighting,
}

fn deserialize_required_response_sha256<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveMachineSettingsDocument {
    schema_version: u32,
    /// Exact selector passed through --input-device (ID or unique full name).
    input_selector: String,
    /// Current caller-declared route; compared with the saved calibration binding before audio access.
    runtime_binding: sotf_capture::live_level::LiveCalibrationBinding,
    /// Optional unweighted RMS/SPL integration band in Hz.
    spl_band_hz: Option<[f64; 2]>,
}

struct LoadedLiveCalibration {
    profile: sotf_capture::live_level::LiveCalibrationProfile,
    machine_settings: sotf_capture::live_level::LiveLevelMachineSettings,
}

fn read_bounded_file(
    path: &Path,
    maximum_bytes: usize,
    description: &str,
) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("cannot open {description} {}: {error}", path.display()))?;
    let read_limit = u64::try_from(maximum_bytes)
        .ok()
        .and_then(|limit| limit.checked_add(1))
        .ok_or_else(|| format!("invalid size limit for {description}"))?;
    let mut limited_file = file.take(read_limit);
    let mut bytes = Vec::new();
    limited_file
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {description} {}: {error}", path.display()))?;
    if bytes.len() > maximum_bytes {
        return Err(format!(
            "{description} {} exceeds the {maximum_bytes}-byte limit",
            path.display()
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = sha2::Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn live_bindings_match(
    saved: &sotf_capture::live_level::LiveCalibrationBinding,
    runtime: &sotf_capture::live_level::LiveCalibrationBinding,
) -> bool {
    saved.microphone_id == runtime.microphone_id
        && saved.host_api == runtime.host_api
        && saved.input_device_id == runtime.input_device_id
        && saved.input_channel == runtime.input_channel
        && saved.sample_rate_hz == runtime.sample_rate_hz
        && saved.input_sample_format == runtime.input_sample_format
        && saved.declared_input_gain_db.to_bits() == runtime.declared_input_gain_db.to_bits()
        && saved.gain_attested == runtime.gain_attested
        && saved.orientation == runtime.orientation
}

fn calibration_pair<'a>(
    calibration_profile: Option<&'a Path>,
    machine_settings: Option<&'a Path>,
) -> Result<Option<(&'a Path, &'a Path)>, String> {
    match (calibration_profile, machine_settings) {
        (Some(profile), Some(settings)) => Ok(Some((profile, settings))),
        (None, None) => Ok(None),
        (Some(_), None) => Err("--calibration-profile requires --machine-settings".into()),
        (None, Some(_)) => Err("--machine-settings requires --calibration-profile".into()),
    }
}

fn load_live_calibration(
    config: &sotf_capture::live_level::LiveLevelConfig,
    calibration_profile_path: &Path,
    machine_settings_path: &Path,
) -> Result<LoadedLiveCalibration, String> {
    use sotf_capture::live_level::{
        LiveCalibrationProfile, LiveCalibrationProfileInput, LiveLevelMachineSettings,
        LiveResponseCurveInput, LiveRmsSplAnchorInput,
    };

    config.validate()?;
    let profile_bytes = read_bounded_file(
        calibration_profile_path,
        LIVE_CALIBRATION_DOCUMENT_MAX_BYTES,
        "calibration profile JSON",
    )?;
    let profile_document: LiveCalibrationDocument = serde_json::from_slice(&profile_bytes)
        .map_err(|error| format!("cannot parse calibration profile JSON: {error}"))?;
    if profile_document.schema_version != LIVE_CALIBRATION_DOCUMENT_VERSION {
        return Err(format!(
            "unsupported calibration profile schema_version {}; expected {}",
            profile_document.schema_version, LIVE_CALIBRATION_DOCUMENT_VERSION
        ));
    }

    let machine_bytes = read_bounded_file(
        machine_settings_path,
        LIVE_CALIBRATION_DOCUMENT_MAX_BYTES,
        "machine settings JSON",
    )?;
    let machine_document: LiveMachineSettingsDocument = serde_json::from_slice(&machine_bytes)
        .map_err(|error| format!("cannot parse machine settings JSON: {error}"))?;
    if machine_document.schema_version != LIVE_CALIBRATION_DOCUMENT_VERSION {
        return Err(format!(
            "unsupported machine settings schema_version {}; expected {}",
            machine_document.schema_version, LIVE_CALIBRATION_DOCUMENT_VERSION
        ));
    }
    if machine_document.input_selector.trim().is_empty()
        || machine_document.input_selector != config.input_device
    {
        return Err("machine input_selector must exactly match --input-device".into());
    }
    if machine_document.runtime_binding.input_channel != config.input_channel {
        return Err("machine binding input_channel must match --input-channel".into());
    }
    if machine_document.runtime_binding.sample_rate_hz != config.sample_rate_hz {
        return Err("machine binding sample_rate_hz must match --sample-rate".into());
    }
    if !live_bindings_match(&profile_document.binding, &machine_document.runtime_binding) {
        return Err(
            "runtime machine binding does not match the saved calibration profile binding".into(),
        );
    }

    let machine_settings = LiveLevelMachineSettings {
        microphone_id: machine_document.runtime_binding.microphone_id.clone(),
        declared_input_gain_db: machine_document.runtime_binding.declared_input_gain_db,
        gain_attested: machine_document.runtime_binding.gain_attested,
        orientation: machine_document.runtime_binding.orientation,
        spl_band_hz: machine_document.spl_band_hz,
    };
    machine_settings.validate()?;
    if machine_settings
        .spl_band_hz
        .is_some_and(|band| band[1] > f64::from(config.sample_rate_hz) / 2.0)
    {
        return Err("machine SPL band must not exceed the configured Nyquist frequency".into());
    }

    let response_curve = profile_document
        .response_curve
        .map(|curve| -> Result<LiveResponseCurveInput, String> {
            if curve.path.as_os_str().is_empty() || curve.path.is_absolute() {
                return Err("response-curve path must be a nonempty relative path".into());
            }
            let profile_directory = calibration_profile_path
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            let curve_path = profile_directory.join(curve.path);
            let bytes =
                read_bounded_file(&curve_path, LIVE_RESPONSE_CURVE_MAX_BYTES, "response curve")?;
            Ok(LiveResponseCurveInput {
                format: curve.format,
                convention: curve.convention,
                reference_frequency_hz: curve.reference_frequency_hz,
                bytes,
                expected_sha256: curve.sha256,
            })
        })
        .transpose()?;
    let rms_anchor = profile_document
        .rms_anchor
        .map(|anchor| LiveRmsSplAnchorInput {
            binding: profile_document.binding.clone(),
            response_sha256: anchor.response_sha256,
            reference_rms_full_scale: anchor.reference_rms_full_scale,
            reference_level_db_spl: anchor.reference_level_db_spl,
            reference_frequency_hz: anchor.reference_frequency_hz,
            band_hz: anchor.band_hz,
            weighting: anchor.weighting,
        });
    let profile = LiveCalibrationProfile::new(LiveCalibrationProfileInput {
        binding: profile_document.binding,
        response_curve,
        spl_anchor: rms_anchor,
    })?;
    Ok(LoadedLiveCalibration {
        profile,
        machine_settings,
    })
}

fn write_json_line(
    output: &mut impl std::io::Write,
    value: &serde_json::Value,
) -> Result<(), String> {
    serde_json::to_writer(&mut *output, value).map_err(|error| error.to_string())?;
    writeln!(output)
        .and_then(|_| output.flush())
        .map_err(|error| error.to_string())
}

fn live_calibration_validation_event(loaded: &LoadedLiveCalibration) -> serde_json::Value {
    serde_json::json!({
        "event": "live_calibration_validated",
        "schema_version": LIVE_CALIBRATION_DOCUMENT_VERSION,
        "calibration_status": "declared_profile_validated_hardware_check_pending",
        "hardware_opened": false,
        "binding": loaded.profile.binding(),
        "response_sha256": loaded.profile.response_sha256(),
        "response_format": loaded.profile.response_format(),
        "response_convention": loaded.profile.response_convention(),
        "response_reference_frequency_hz": loaded.profile.response_reference_frequency_hz(),
        "has_spl_anchor": loaded.profile.has_spl_anchor(),
        "machine_settings": &loaded.machine_settings,
    })
}

fn run_live_monitor(
    config: sotf_capture::live_level::LiveLevelConfig,
    calibration_profile_path: Option<&Path>,
    machine_settings_path: Option<&Path>,
    validate_calibration: bool,
) -> Result<(), String> {
    use sotf_capture::live_level::{stream_live_levels, stream_live_levels_calibrated};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    config.validate()?;
    let calibration = calibration_pair(calibration_profile_path, machine_settings_path)?
        .map(|(profile_path, settings_path)| {
            load_live_calibration(&config, profile_path, settings_path)
        })
        .transpose()?;
    if validate_calibration {
        let loaded = calibration.as_ref().ok_or_else(|| {
            "--validate-calibration requires a calibration profile/settings pair".to_owned()
        })?;
        let mut output = std::io::stdout().lock();
        let event = live_calibration_validation_event(loaded);
        return write_json_line(&mut output, &event);
    }

    let cancel = Arc::new(AtomicBool::new(false));
    let handler_cancel = Arc::clone(&cancel);
    ctrlc::set_handler(move || handler_cancel.store(true, Ordering::Relaxed))
        .map_err(|error| format!("cannot install monitoring cancellation handler: {error}"))?;
    let mut output = std::io::stdout().lock();
    let summary = match calibration.as_ref() {
        Some(loaded) => stream_live_levels_calibrated(
            &config,
            &loaded.machine_settings,
            &loaded.profile,
            &cancel,
            |frame| {
                write_json_line(
                    &mut output,
                    &serde_json::json!({"event":"live_frame", "frame": frame}),
                )
            },
        )?,
        None => stream_live_levels(&config, &cancel, |frame| {
            write_json_line(
                &mut output,
                &serde_json::json!({"event":"live_frame", "frame": frame}),
            )
        })?,
    };
    write_json_line(
        &mut output,
        &serde_json::json!({"event":"live_stopped", "summary":summary}),
    )
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Command::Devices { json } => list_audio_devices(json),
        Command::Live {
            input_device,
            input_channel,
            sample_rate,
            fft_size,
            duration,
            calibration_profile,
            machine_settings,
            validate_calibration,
        } => run_live_monitor(
            sotf_capture::live_level::LiveLevelConfig {
                input_device,
                input_channel,
                sample_rate_hz: sample_rate,
                fft_size,
                duration_secs: duration,
            },
            calibration_profile.as_deref(),
            machine_settings.as_deref(),
            validate_calibration,
        ),
        Command::Capture {
            signal,
            duration,
            sample_rate,
            channels,
            hwaudio_send_to,
            hwaudio_record_from,
            name,
            output_dir,
            device,
            input_device,
            output_device,
            freq,
            freq1,
            freq2,
            start_freq,
            end_freq,
            amp,
            amp1,
            amp2,
            mls_order,
            microphone_compensation,
            mic_calibrations,
            json,
        } => record_signal(
            signal,
            duration,
            sample_rate,
            channels,
            hwaudio_send_to,
            hwaudio_record_from,
            name,
            output_dir,
            device,
            input_device,
            output_device,
            freq,
            freq1,
            freq2,
            start_freq,
            end_freq,
            amp,
            amp1,
            amp2,
            mls_order,
            microphone_compensation,
            parse_mic_calibrations(&mic_calibrations)?,
            json,
        ),
        Command::PlanValidate { plan } => plan_validate(&plan),
        Command::PlanRecord { plan, output } => plan_record(&plan, &output),
        Command::PlanProcess {
            raw,
            output,
            selected_take_ids,
        } => plan_process(&raw, &output, &selected_take_ids),
    }
}

fn main() {
    env_logger::init();
    if let Err(error) = run() {
        eprintln!("sotf-capture: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_live_config(input_device: &str) -> sotf_capture::live_level::LiveLevelConfig {
        sotf_capture::live_level::LiveLevelConfig {
            input_device: input_device.into(),
            input_channel: 0,
            sample_rate_hz: 48_000,
            fft_size: 4096,
            duration_secs: 1.0,
        }
    }

    fn write_calibration_fixture(
        directory: &Path,
        response_curve: &[u8],
        expected_sha256: Option<&str>,
        input_selector: &str,
        input_channel: u16,
        sample_rate_hz: u32,
    ) -> (PathBuf, PathBuf) {
        let profile_path = directory.join("calibration.json");
        let machine_path = directory.join("machine.json");
        std::fs::write(directory.join("response.csv"), response_curve).unwrap();
        let binding = json!({
            "microphone_id": "synthetic-cli-test-mic",
            "host_api": "SyntheticHost",
            "input_device_id": "synthetic-device-id",
            "input_channel": input_channel,
            "sample_rate_hz": sample_rate_hz,
            "input_sample_format": "F32",
            "declared_input_gain_db": 0.0,
            "gain_attested": true,
            "orientation": "on_axis",
        });
        let profile = json!({
            "schema_version": 1,
            "binding": binding,
            "response_curve": {
                "path": "response.csv",
                "format": "csv",
                "convention": "positive_db_means_microphone_too_loud",
                "reference_frequency_hz": 1000.0,
                "sha256": expected_sha256,
            },
            "rms_anchor": {
                "response_sha256": sha256_hex(response_curve),
                "reference_rms_full_scale": 0.1,
                "reference_level_db_spl": 94.0,
                "reference_frequency_hz": 1000.0,
                "band_hz": [990.0, 1010.0],
                "weighting": "z",
            },
        });
        let machine = json!({
            "schema_version": 1,
            "input_selector": input_selector,
            "runtime_binding": binding,
            "spl_band_hz": [100.0, 20000.0],
        });
        std::fs::write(&profile_path, serde_json::to_vec(&profile).unwrap()).unwrap();
        std::fs::write(&machine_path, serde_json::to_vec(&machine).unwrap()).unwrap();
        (profile_path, machine_path)
    }

    #[test]
    fn capture_accepts_independent_machine_device_selectors() {
        let cli = Cli::try_parse_from([
            "sotf-capture",
            "capture",
            "--hwaudio-send-to",
            "2",
            "--hwaudio-record-from",
            "0",
            "--input-device",
            "mic-id",
            "--output-device",
            "interface-id",
            "--mic-calibration",
            "0=calibration.txt",
        ])
        .unwrap();
        match cli.command {
            Command::Capture {
                input_device,
                output_device,
                device,
                mic_calibrations,
                ..
            } => {
                assert_eq!(input_device.as_deref(), Some("mic-id"));
                assert_eq!(output_device.as_deref(), Some("interface-id"));
                assert!(device.is_none());
                assert_eq!(
                    parse_mic_calibrations(&mic_calibrations).unwrap()[&0],
                    "calibration.txt"
                );
            }
            _ => panic!("expected capture command"),
        }
    }

    #[test]
    fn shared_device_selector_refuses_conflicting_independent_selector() {
        for flag in ["--input-device", "--output-device"] {
            assert!(
                Cli::try_parse_from([
                    "sotf-capture",
                    "capture",
                    "--hwaudio-send-to",
                    "0",
                    "--hwaudio-record-from",
                    "0",
                    "--device",
                    "shared",
                    flag,
                    "other",
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn plan_process_accepts_one_explicit_flag_per_selected_take() {
        let cli = Cli::try_parse_from([
            "sotf-capture",
            "plan-process",
            "--raw",
            "session/capture-raw.json",
            "--output",
            "processed",
            "--select-take",
            "take-r000-0123456789abcdef",
            "--select-take",
            "take-r000-fedcba9876543210",
        ])
        .unwrap();
        match cli.command {
            Command::PlanProcess {
                raw,
                output,
                selected_take_ids,
            } => {
                assert_eq!(raw, PathBuf::from("session/capture-raw.json"));
                assert_eq!(output, PathBuf::from("processed"));
                assert_eq!(selected_take_ids.len(), 2);
                assert_eq!(selected_take_ids[0], "take-r000-0123456789abcdef");
                assert_eq!(selected_take_ids[1], "take-r000-fedcba9876543210");
            }
            _ => panic!("expected plan-process command"),
        }
    }

    #[test]
    fn calibration_channel_assignments_refuse_ambiguity() {
        for values in [
            vec!["0=a", "0=b"],
            vec!["64=a"],
            vec!["0="],
            vec!["0=  "],
            vec!["bad=a"],
        ] {
            let values: Vec<_> = values.into_iter().map(str::to_owned).collect();
            assert!(parse_mic_calibrations(&values).is_err(), "{values:?}");
        }
    }

    #[test]
    fn live_cli_requires_calibration_profile_and_machine_settings_as_a_pair() {
        let base = ["sotf-capture", "live", "--input-device", "device"];
        assert!(
            Cli::try_parse_from([&base[..], &["--calibration-profile", "profile.json"],].concat())
                .is_err()
        );
        assert!(
            Cli::try_parse_from([&base[..], &["--machine-settings", "machine.json"],].concat())
                .is_err()
        );
        assert!(Cli::try_parse_from([&base[..], &["--validate-calibration"],].concat()).is_err());
    }

    #[test]
    fn live_calibration_dry_run_loads_relative_curve_and_never_opens_audio() {
        let directory = tempfile::tempdir().unwrap();
        let curve = b"20,-1\n1000,0\n20000,1\n";
        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            curve,
            None,
            "synthetic-device-id",
            0,
            48_000,
        );
        let config = test_live_config("synthetic-device-id");
        let loaded = load_live_calibration(&config, &profile_path, &machine_path).unwrap();
        assert_eq!(loaded.profile.response_bytes(), Some(curve.as_slice()));
        assert_eq!(
            loaded.profile.response_sha256(),
            Some(sha256_hex(curve).as_str())
        );
        assert!(loaded.profile.has_spl_anchor());
        assert_eq!(loaded.machine_settings.spl_band_hz, Some([100.0, 20_000.0]));
        let validation_event = live_calibration_validation_event(&loaded);
        assert_eq!(validation_event["schema_version"], 1);
        assert_eq!(
            validation_event["calibration_status"],
            "declared_profile_validated_hardware_check_pending"
        );
        assert_eq!(validation_event["hardware_opened"].as_bool(), Some(false));

        // The synthetic device selector is not present; success proves dry-run returns before device enumeration.
        run_live_monitor(config, Some(&profile_path), Some(&machine_path), true).unwrap();
    }

    #[test]
    fn live_calibration_refuses_malformed_or_oversized_json_and_curve_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            b"20,-1\n1000,0\n20000,1\n",
            None,
            "synthetic-device-id",
            0,
            48_000,
        );
        let config = test_live_config("synthetic-device-id");

        std::fs::write(&profile_path, b"{ malformed json").unwrap();
        assert!(load_live_calibration(&config, &profile_path, &machine_path).is_err());

        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            b"20,-1\n1000,0\n20000,1\n",
            None,
            "synthetic-device-id",
            0,
            48_000,
        );
        std::fs::write(&machine_path, b"{ malformed json").unwrap();
        assert!(load_live_calibration(&config, &profile_path, &machine_path).is_err());

        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            b"20,-1\n1000,0\n20000,1\n",
            None,
            "synthetic-device-id",
            0,
            48_000,
        );
        std::fs::write(
            &machine_path,
            vec![b' '; LIVE_CALIBRATION_DOCUMENT_MAX_BYTES + 1],
        )
        .unwrap();
        let error = load_live_calibration(&config, &profile_path, &machine_path)
            .err()
            .unwrap();
        assert!(error.contains("machine settings JSON") && error.contains("exceeds the"));

        std::fs::write(
            &profile_path,
            vec![b' '; LIVE_CALIBRATION_DOCUMENT_MAX_BYTES + 1],
        )
        .unwrap();
        let error = load_live_calibration(&config, &profile_path, &machine_path)
            .err()
            .unwrap();
        assert!(error.contains("exceeds the"));

        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            &vec![b' '; LIVE_RESPONSE_CURVE_MAX_BYTES + 1],
            None,
            "synthetic-device-id",
            0,
            48_000,
        );
        let error = load_live_calibration(&config, &profile_path, &machine_path)
            .err()
            .unwrap();
        assert!(error.contains("response curve") && error.contains("exceeds the"));
    }

    #[test]
    fn live_calibration_refuses_machine_route_conflicts_and_response_digest_mismatch() {
        let directory = tempfile::tempdir().unwrap();
        let curve = b"20,-1\n1000,0\n20000,1\n";
        let (profile_path, machine_path) =
            write_calibration_fixture(directory.path(), curve, None, "other-selector", 0, 48_000);
        let config = test_live_config("synthetic-device-id");
        let error = load_live_calibration(&config, &profile_path, &machine_path)
            .err()
            .unwrap();
        assert!(error.contains("input_selector"));

        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            curve,
            None,
            "synthetic-device-id",
            1,
            48_000,
        );
        let error = load_live_calibration(&config, &profile_path, &machine_path)
            .err()
            .unwrap();
        assert!(error.contains("input_channel"));

        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            curve,
            None,
            "synthetic-device-id",
            0,
            44_100,
        );
        let error = load_live_calibration(&config, &profile_path, &machine_path)
            .err()
            .unwrap();
        assert!(error.contains("sample_rate_hz"));

        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            curve,
            Some(&"0".repeat(64)),
            "synthetic-device-id",
            0,
            48_000,
        );
        let error = load_live_calibration(&config, &profile_path, &machine_path)
            .err()
            .unwrap();
        assert!(error.contains("SHA-256"));
    }

    #[test]
    fn live_calibration_does_not_rebind_saved_anchor_to_runtime_gain_or_device() {
        let directory = tempfile::tempdir().unwrap();
        let curve = b"20,-1\n1000,0\n20000,1\n";
        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            curve,
            None,
            "synthetic-device-id",
            0,
            48_000,
        );
        let config = test_live_config("synthetic-device-id");
        let original_machine: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&machine_path).unwrap()).unwrap();

        for (field, replacement) in [
            ("declared_input_gain_db", json!(6.0)),
            ("input_device_id", json!("different-runtime-device")),
        ] {
            let mut changed_machine = original_machine.clone();
            changed_machine["runtime_binding"][field] = replacement;
            std::fs::write(&machine_path, serde_json::to_vec(&changed_machine).unwrap()).unwrap();

            let error = load_live_calibration(&config, &profile_path, &machine_path)
                .err()
                .unwrap();
            assert!(
                error.contains("runtime machine binding does not match the saved calibration"),
                "field {field}: {error}"
            );
        }
    }

    #[test]
    fn live_calibration_anchor_digest_rejects_changed_curve_without_optional_curve_digest() {
        let directory = tempfile::tempdir().unwrap();
        let original_curve = b"20,-1\n1000,0\n20000,1\n";
        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            original_curve,
            None,
            "synthetic-device-id",
            0,
            48_000,
        );
        let config = test_live_config("synthetic-device-id");
        let profile_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&profile_path).unwrap()).unwrap();
        assert!(profile_json["response_curve"]["sha256"].is_null());
        assert_eq!(
            profile_json["rms_anchor"]["response_sha256"],
            json!(sha256_hex(original_curve))
        );

        std::fs::write(
            directory.path().join("response.csv"),
            b"20,-2\n1000,0\n20000,1\n",
        )
        .unwrap();
        let error = load_live_calibration(&config, &profile_path, &machine_path)
            .err()
            .unwrap();
        assert!(error.contains("anchor response hash"), "{error}");
    }

    #[test]
    fn live_calibration_requires_explicit_anchor_response_digest_field() {
        let directory = tempfile::tempdir().unwrap();
        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            b"20,-1\n1000,0\n20000,1\n",
            None,
            "synthetic-device-id",
            0,
            48_000,
        );
        let mut profile_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&profile_path).unwrap()).unwrap();
        let removed_digest = profile_json["rms_anchor"]
            .as_object_mut()
            .unwrap()
            .remove("response_sha256");
        assert!(
            removed_digest.is_some(),
            "fixture must contain the anchor digest"
        );
        std::fs::write(&profile_path, serde_json::to_vec(&profile_json).unwrap()).unwrap();

        let error = load_live_calibration(
            &test_live_config("synthetic-device-id"),
            &profile_path,
            &machine_path,
        )
        .err()
        .unwrap();
        assert!(error.contains("response_sha256"), "{error}");
    }

    #[test]
    fn live_calibration_accepts_null_anchor_digest_only_without_a_response_curve() {
        let directory = tempfile::tempdir().unwrap();
        let (profile_path, machine_path) = write_calibration_fixture(
            directory.path(),
            b"20,-1\n1000,0\n20000,1\n",
            None,
            "synthetic-device-id",
            0,
            48_000,
        );
        let mut profile_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&profile_path).unwrap()).unwrap();
        profile_json
            .as_object_mut()
            .unwrap()
            .remove("response_curve");
        profile_json["rms_anchor"]["response_sha256"] = serde_json::Value::Null;
        std::fs::write(&profile_path, serde_json::to_vec(&profile_json).unwrap()).unwrap();

        let loaded = load_live_calibration(
            &test_live_config("synthetic-device-id"),
            &profile_path,
            &machine_path,
        )
        .unwrap();
        assert!(loaded.profile.has_spl_anchor());
        assert_eq!(loaded.profile.response_bytes(), None);
        assert_eq!(loaded.profile.response_sha256(), None);
    }
}
