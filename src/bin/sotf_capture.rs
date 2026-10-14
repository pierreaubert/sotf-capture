//! Standalone capture CLI over the sotf-capture library.
//!
//! Lists audio devices, captures swept-sine (or other stimulus) takes
//! through the cpal-native playback backend, and records multi-source
//! capture-session plans. Every command mirrors the corresponding
//! `sotf` frontend flow so behavior stays identical across shells.

// Rust guideline compliant 2026-02-21

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::PathBuf;

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
    },
    /// Capture every microphone simultaneously for each sequential output.
    Capture {
        /// Stimulus: sweep, tone, two-tone, white-noise, pink-noise, stipa,
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
        /// Hardware microphone inputs recorded together for every output, e.g. "0,1,2,3".
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
    /// Probe per-channel arrival times with sequential multisine bursts.
    Probe {
        /// Hardware output channels, e.g. "0,1".
        #[arg(long)]
        hwaudio_send_to: String,
        /// Channel names, e.g. "L,R"; defaults to ch{N} labels.
        #[arg(long)]
        channel_names: Option<String>,
        /// Single hardware input channel, e.g. "0".
        #[arg(long)]
        hwaudio_record_from: String,
        /// Sample rate in Hz.
        #[arg(long, default_value_t = 48000)]
        sample_rate: u32,
        /// Probe duration per channel in milliseconds.
        #[arg(long, default_value_t = 1000.0)]
        probe_duration_ms: f32,
        /// Silence gap between probes in milliseconds.
        #[arg(long, default_value_t = 500.0)]
        silence_duration_ms: f32,
        /// Probe playback level in dBFS.
        #[arg(long, default_value_t = -6.0206, allow_negative_numbers = true)]
        signal_level_db: f32,
        /// Output filename prefix.
        #[arg(long)]
        name: Option<String>,
        /// Output directory (created when missing).
        #[arg(long)]
        output_dir: Option<PathBuf>,
        /// Shared input/output device selector (legacy convenience).
        #[arg(long, conflicts_with_all = ["input_device", "output_device"])]
        device: Option<String>,
        /// Independent input device ID or unique name.
        #[arg(long)]
        input_device: Option<String>,
        /// Independent output device ID or unique name.
        #[arg(long)]
        output_device: Option<String>,
        /// Emit the results as JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Capture per-channel low-frequency phase with steady-state tones.
    BassAnchor {
        /// Hardware output channels, e.g. "0,1".
        #[arg(long)]
        hwaudio_send_to: String,
        /// Channel names, e.g. "L,R"; defaults to ch{N} labels.
        #[arg(long)]
        channel_names: Option<String>,
        /// Single hardware input channel, e.g. "0".
        #[arg(long)]
        hwaudio_record_from: String,
        /// Loopback input channel on the recording device, when wired.
        #[arg(long)]
        loopback_from: Option<u16>,
        /// Sample rate in Hz.
        #[arg(long, default_value_t = 48000)]
        sample_rate: u32,
        /// Tone centre frequency in Hz.
        #[arg(long, default_value_t = 30.0)]
        bass_freq_hz: f32,
        /// Tone length in seconds, steady portion plus fades.
        #[arg(long, default_value_t = 2.0)]
        bass_duration_s: f32,
        /// Half-Hann fade length in milliseconds.
        #[arg(long, default_value_t = 50.0)]
        fade_ms: f32,
        /// Lock-in sub-window count.
        #[arg(long, default_value_t = 8)]
        num_windows: u16,
        /// Silence gap between tones in milliseconds.
        #[arg(long, default_value_t = 500.0)]
        silence_duration_ms: f32,
        /// Tone playback level in dBFS.
        #[arg(long, default_value_t = -6.0206, allow_negative_numbers = true)]
        signal_level_db: f32,
        /// Output filename prefix.
        #[arg(long)]
        name: Option<String>,
        /// Output directory (created when missing).
        #[arg(long)]
        output_dir: Option<PathBuf>,
        /// Shared input/output device selector (legacy convenience).
        #[arg(long, conflicts_with_all = ["input_device", "output_device"])]
        device: Option<String>,
        /// Independent input device ID or unique name.
        #[arg(long)]
        input_device: Option<String>,
        /// Independent output device ID or unique name.
        #[arg(long)]
        output_device: Option<String>,
        /// Emit the results as JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Merge per-position CTC fragments into one self-contained bundle.
    PlanCtcMerge {
        /// Processed session directories, one head position each.
        #[arg(long = "session", required = true)]
        sessions: Vec<PathBuf>,
        /// New output directory for the merged bundle.
        #[arg(long)]
        output: PathBuf,
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
    ///
    /// Single-plan discipline: subs and mains for joint optimization
    /// must come from one plan/process run with identical response
    /// grids, or downstream GD-Opt skips joint optimization.
    PlanProcess {
        /// Path to the immutable capture-raw.json journal.
        #[arg(long)]
        raw: PathBuf,
        /// New output directory for processed evidence and optional `RoomEQ` projection.
        #[arg(long)]
        output: PathBuf,
        /// Explicit take ID to project; repeat once per source/microphone pair.
        #[arg(long = "select-take")]
        selected_take_ids: Vec<String>,
        /// Microphone backing measured-IR declarations; the first plan
        /// microphone when absent.
        #[arg(long)]
        ir_microphone: Option<String>,
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
#[expect(
    clippy::too_many_lines,
    reason = "The CLI dispatch keeps each command and its explicit flag mapping together."
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "CLI dispatch transfers ownership of parsed flags to the command handler."
)]
fn record_signal(
    playback: &mut impl sotf_capture::signal_recorder::SweepPlayback,
    recorder: &mut impl sotf_capture::signal_recorder::SweepRecorder,
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
        DEFAULT_MLS_ORDER, SignalParams, SignalType, generate_output_filenames_stereo,
        generate_signal, parse_channel_list, prepare_measurement_signal, prepare_signal,
        record_and_analyze_with_multi_recorder, validate_signal_params, write_temp_wav,
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
    if record_from_channels.is_empty() {
        return Err("hwaudio-record-from must specify at least 1 microphone channel".into());
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
        // STIPA carries a standardized spectrum and level; --amp does not
        // apply and any supplied value is ignored like other unused flags.
        SignalType::Stipa => SignalParams::Stipa,
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
        let (wav_paths, csv_paths): (Vec<_>, Vec<_>) = record_from_channels
            .iter()
            .map(|&record_ch| {
                let (wav_name, csv_name) = generate_output_filenames_stereo(
                    name.as_deref(),
                    signal_type,
                    send_ch,
                    record_ch,
                    sample_rate,
                );
                (output_dir.join(wav_name), output_dir.join(csv_name))
            })
            .unzip();
        let temp_wav = write_temp_wav(&prepared_signal, sample_rate, 1)?;
        let mic_calibrations: Vec<_> = record_from_channels
            .iter()
            .map(|channel| {
                mic_calibration_map
                    .get(&usize::from(*channel))
                    .or(microphone_compensation.as_ref())
                    .cloned()
            })
            .collect();
        let captures = record_and_analyze_with_multi_recorder(
            playback,
            recorder,
            temp_wav.path(),
            &wav_paths,
            &prepared_signal,
            sample_rate,
            &csv_paths,
            send_ch,
            &record_from_channels,
            output_device.as_deref().or(device.as_deref()),
            input_device.as_deref().or(device.as_deref()),
            &mic_calibrations,
            (signal_type == SignalType::Sweep).then_some((start_freq, end_freq)),
            1,
            None,
        )?;
        for (mic_index, (&record_ch, capture)) in
            record_from_channels.iter().zip(&captures).enumerate()
        {
            let wav_path = &wav_paths[mic_index];
            let csv_path = &csv_paths[mic_index];
            let effective_mic_comp = mic_calibrations[mic_index].as_deref();
            let quality = summarize_take_quality(capture);
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
        }
        if index + 1 < send_to_channels.len() {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
    Ok(())
}

fn resolve_aux_channels(
    send_to: &str,
    record_from: &str,
    channel_names: Option<&str>,
) -> Result<(Vec<u16>, Vec<String>, u16), String> {
    use sotf_capture::signal_recorder::parse_channel_list;

    let send = parse_channel_list(send_to)?;
    if send.is_empty() {
        return Err("hwaudio-send-to must specify at least 1 channel".to_string());
    }
    let record = parse_channel_list(record_from)?;
    if record.len() != 1 {
        return Err("hwaudio-record-from must specify exactly 1 channel".to_string());
    }
    let names = match channel_names {
        None => send.iter().map(|channel| format!("ch{channel}")).collect(),
        Some(list) => {
            let names: Vec<String> = list.split(',').map(|name| name.trim().to_owned()).collect();
            if names.len() != send.len() || names.iter().any(String::is_empty) {
                return Err(
                    "channel-names must name every send-to channel exactly once".to_string()
                );
            }
            names
        }
    };
    Ok((send, names, record[0]))
}

fn check_device_selectors(
    device: Option<&str>,
    input_device: Option<&str>,
    output_device: Option<&str>,
) -> Result<(), String> {
    for selector in [device, input_device, output_device].into_iter().flatten() {
        if selector.trim().is_empty() {
            return Err("device selectors must be nonempty when supplied".into());
        }
    }
    Ok(())
}

fn aux_output_dir(output_dir: Option<PathBuf>) -> Result<PathBuf, String> {
    let output_dir = match output_dir {
        Some(dir) => dir,
        None => std::env::current_dir()
            .map_err(|e| format!("cannot determine current directory: {e}"))?,
    };
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("failed to create output directory: {e}"))?;
    Ok(output_dir)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Clap hands each flag through once; a parameter object would only rename the plumbing"
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "CLI dispatch transfers ownership of parsed flags to the command handler."
)]
fn run_probe(
    hwaudio_send_to: String,
    channel_names: Option<String>,
    hwaudio_record_from: String,
    sample_rate: u32,
    probe_duration_ms: f32,
    silence_duration_ms: f32,
    signal_level_db: f32,
    name: Option<String>,
    output_dir: Option<PathBuf>,
    device: Option<String>,
    input_device: Option<String>,
    output_device: Option<String>,
    json: bool,
) -> Result<(), String> {
    use sotf_capture::signal_recorder::probe_channel_delays_with_recording;

    let output_dir = aux_output_dir(output_dir)?;
    check_device_selectors(
        device.as_deref(),
        input_device.as_deref(),
        output_device.as_deref(),
    )?;
    let (send, names, record) = resolve_aux_channels(
        &hwaudio_send_to,
        &hwaudio_record_from,
        channel_names.as_deref(),
    )?;
    let prefix = name.unwrap_or_else(|| "probe".into());
    let wav_path = output_dir.join(format!("{prefix}-probe.wav"));
    let results_path = output_dir.join(format!("{prefix}-probe-results.json"));
    let results = probe_channel_delays_with_recording(
        &send,
        &names,
        sample_rate,
        probe_duration_ms,
        silence_duration_ms,
        output_device.as_deref().or(device.as_deref()),
        input_device.as_deref().or(device.as_deref()),
        record,
        &wav_path,
        signal_level_db,
        None,
    )?;
    std::fs::write(
        &results_path,
        serde_json::to_string_pretty(&results).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("cannot write probe results: {e}"))?;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "version": 1,
                "wav": wav_path.display().to_string(),
                "results": results_path.display().to_string(),
                "delays": results,
            })
        );
        return Ok(());
    }
    println!("probe recording: {}", wav_path.display());
    for (channel, delay) in results
        .channels
        .iter()
        .zip(results.alignment_delays_ms.iter())
    {
        println!(
            "  {}: arrival={:.3} ms gain={:.1} dB snr={:.1} dB align={:.3} ms",
            channel.channel_name, channel.arrival_ms, channel.gain_db, channel.snr_db, delay
        );
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Clap hands each flag through once; a parameter object would only rename the plumbing"
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "CLI dispatch transfers ownership of parsed flags to the command handler."
)]
fn run_bass_anchor(
    hwaudio_send_to: String,
    channel_names: Option<String>,
    hwaudio_record_from: String,
    loopback_from: Option<u16>,
    sample_rate: u32,
    bass_freq_hz: f32,
    bass_duration_s: f32,
    fade_ms: f32,
    num_windows: u16,
    silence_duration_ms: f32,
    signal_level_db: f32,
    name: Option<String>,
    output_dir: Option<PathBuf>,
    device: Option<String>,
    input_device: Option<String>,
    output_device: Option<String>,
    json: bool,
) -> Result<(), String> {
    use sotf_capture::signal_recorder::run_bass_anchor_with_recording;

    let output_dir = aux_output_dir(output_dir)?;
    check_device_selectors(
        device.as_deref(),
        input_device.as_deref(),
        output_device.as_deref(),
    )?;
    let (send, names, record) = resolve_aux_channels(
        &hwaudio_send_to,
        &hwaudio_record_from,
        channel_names.as_deref(),
    )?;
    if loopback_from == Some(record) {
        return Err("loopback-from must differ from the recording channel".into());
    }
    let prefix = name.unwrap_or_else(|| "bass-anchor".into());
    let wav_path = output_dir.join(format!("{prefix}-bass-anchor.wav"));
    let results_path = output_dir.join(format!("{prefix}-bass-anchor-results.json"));
    let results = run_bass_anchor_with_recording(
        &send,
        &names,
        sample_rate,
        bass_freq_hz,
        bass_duration_s,
        fade_ms,
        num_windows,
        silence_duration_ms,
        output_device.as_deref().or(device.as_deref()),
        input_device.as_deref().or(device.as_deref()),
        record,
        loopback_from,
        &wav_path,
        signal_level_db,
        None,
    )?;
    std::fs::write(
        &results_path,
        serde_json::to_string_pretty(&results).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("cannot write bass-anchor results: {e}"))?;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "version": 1,
                "wav": wav_path.display().to_string(),
                "results": results_path.display().to_string(),
                "anchors": results,
            })
        );
        return Ok(());
    }
    println!("bass-anchor recording: {}", wav_path.display());
    for channel in &results.channels {
        println!(
            "  {}: phase={:.2} deg magnitude={:.4} stability={:.2} deg{}",
            channel.channel_name,
            channel.bass_anchor_phase_deg,
            channel.bass_anchor_magnitude,
            channel.bass_anchor_stability_deg,
            if channel.bass_anchor_stability_deg < 20.0 {
                ""
            } else {
                " UNRELIABLE"
            }
        );
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
    let plan_directory = plan_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut failures = Vec::new();
    for mic in &session.plan().microphones {
        let path = plan_directory.join(&mic.calibration_file);
        let check = std::fs::File::open(&path).and_then(|file| {
            if file.metadata()?.is_file() {
                Ok(())
            } else {
                Err(std::io::Error::other("not a regular file"))
            }
        });
        if let Err(error) = check {
            failures.push(format!("{}: {} — {error}", mic.id, path.display()));
        }
    }
    if !failures.is_empty() {
        return Err(format!(
            "calibration file validation failed (relative paths use the plan directory):\n{}",
            failures.join("\n")
        ));
    }
    if let Some(bench) = &session.plan().localization_calibration_file {
        let path = plan_directory.join(bench);
        if let Err(error) =
            sotf_capture::capture_session::localization::load_bench_calibration(&path)
        {
            return Err(format!(
                "localization bench calibration invalid (relative paths use the plan directory): {error}"
            ));
        }
    }
    println!(
        "Session plan valid: {:?}, {} microphones, {} sources, {} Hz.",
        session.plan().geometry,
        session.plan().microphones.len(),
        session.plan().sources.len(),
        session.plan().sample_rate_hz,
    );
    println!(
        "{}",
        sotf_capture::capture_session::timing_emitter_summary(&session)
    );
    for line in sotf_capture::capture_session::localization::localization_readiness(session.plan())
    {
        println!("{line}");
    }
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
    ir_microphone: Option<String>,
) -> Result<(), String> {
    use sotf_capture::capture_session::clock::io::{
        ProcessOptions, process_capture_session_with_options,
    };

    if raw_manifest_path.file_name().and_then(|name| name.to_str()) != Some("capture-raw.json") {
        return Err("--raw must name a capture-raw.json journal".into());
    }
    if ir_microphone.as_deref().is_some_and(str::is_empty) {
        return Err("--ir-microphone must be nonempty when supplied".into());
    }
    let raw_directory = raw_manifest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let report = process_capture_session_with_options(
        raw_directory,
        output,
        &ProcessOptions {
            selected_take_ids: selected_take_ids.to_vec(),
            ir_microphone,
        },
    )?;
    for stage in &report.pending_processing {
        if let Some(warning) = stage.strip_prefix("joint_optimization:") {
            eprintln!("warning: joint optimization blocked:{warning}");
        }
    }
    println!(
        "{}",
        sotf_capture::capture_session::clock::io::timing_reference_summary(&report)
    );
    for line in sotf_capture::capture_session::clock::io::clock_skew_lines(&report) {
        println!("{line}");
    }
    if let Some(localization) = report.localization_report.as_ref() {
        for line in
            sotf_capture::capture_session::localization::localization_summary_lines(localization)
        {
            println!("{line}");
        }
        for issue in &localization.issues {
            eprintln!("warning: localization: {issue}");
        }
    }
    for warning in sotf_capture::capture_session::clock::io::clock_skew_warnings(&report) {
        eprintln!("warning: clock skew: {warning}");
    }
    if let Some(ctc) = &report.ctc_measurements {
        println!(
            "CTC raw-sweep fragment: {} ({} files).",
            ctc,
            report.ctc_files.len()
        );
    }
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

#[expect(
    clippy::too_many_lines,
    reason = "The CLI dispatch keeps each command and its explicit flag mapping together."
)]
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
        } => {
            use sotf_capture::live_level::{LiveLevelConfig, stream_live_levels};
            use std::io::Write;
            use std::sync::Arc;
            use std::sync::atomic::{AtomicBool, Ordering};
            let config = LiveLevelConfig {
                input_device,
                input_channel,
                sample_rate_hz: sample_rate,
                fft_size,
                duration_secs: duration,
            };
            config.validate()?;
            let cancel = Arc::new(AtomicBool::new(false));
            let handler_cancel = Arc::clone(&cancel);
            ctrlc::set_handler(move || handler_cancel.store(true, Ordering::Relaxed)).map_err(
                |error| format!("cannot install monitoring cancellation handler: {error}"),
            )?;
            let mut output = std::io::stdout().lock();
            let summary = stream_live_levels(&config, &cancel, |frame| {
                serde_json::to_writer(
                    &mut output,
                    &serde_json::json!({"event":"live_frame", "frame": frame}),
                )
                .map_err(|error| error.to_string())?;
                writeln!(output)
                    .and_then(|()| output.flush())
                    .map_err(|error| error.to_string())
            })?;
            serde_json::to_writer(
                &mut output,
                &serde_json::json!({"event":"live_stopped", "summary":summary}),
            )
            .map_err(|error| error.to_string())?;
            writeln!(output).map_err(|error| error.to_string())
        }
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
            &mut sotf_capture::signal_recorder::CpalPlayback::new(),
            &mut sotf_capture::signal_recorder::CpalRecorder::new("record_and_analyze"),
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
        Command::Probe {
            hwaudio_send_to,
            channel_names,
            hwaudio_record_from,
            sample_rate,
            probe_duration_ms,
            silence_duration_ms,
            signal_level_db,
            name,
            output_dir,
            device,
            input_device,
            output_device,
            json,
        } => run_probe(
            hwaudio_send_to,
            channel_names,
            hwaudio_record_from,
            sample_rate,
            probe_duration_ms,
            silence_duration_ms,
            signal_level_db,
            name,
            output_dir,
            device,
            input_device,
            output_device,
            json,
        ),
        Command::BassAnchor {
            hwaudio_send_to,
            channel_names,
            hwaudio_record_from,
            loopback_from,
            sample_rate,
            bass_freq_hz,
            bass_duration_s,
            fade_ms,
            num_windows,
            silence_duration_ms,
            signal_level_db,
            name,
            output_dir,
            device,
            input_device,
            output_device,
            json,
        } => run_bass_anchor(
            hwaudio_send_to,
            channel_names,
            hwaudio_record_from,
            loopback_from,
            sample_rate,
            bass_freq_hz,
            bass_duration_s,
            fade_ms,
            num_windows,
            silence_duration_ms,
            signal_level_db,
            name,
            output_dir,
            device,
            input_device,
            output_device,
            json,
        ),
        Command::PlanCtcMerge { sessions, output } => {
            let merged =
                sotf_capture::capture_session::ctc::merge_ctc_fragments(&sessions, &output)?;
            println!(
                "Merged CTC bundle in {}: {} head positions, {} speakers, {} files.",
                output.display(),
                merged.heads.len(),
                merged.speakers.len(),
                merged.files.len(),
            );
            Ok(())
        }
        Command::PlanValidate { plan } => plan_validate(&plan),
        Command::PlanRecord { plan, output } => plan_record(&plan, &output),
        Command::PlanProcess {
            raw,
            output,
            selected_take_ids,
            ir_microphone,
        } => plan_process(&raw, &output, &selected_take_ids, ir_microphone),
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
#[expect(
    clippy::cast_possible_truncation,
    reason = "Synthetic fixtures use bounded sample counts and intentionally quantized audio; exact comparisons verify unchanged settings and samples."
)]
mod tests {
    use super::*;

    #[cfg(feature = "test-util")]
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "This end-to-end fixture keeps setup, execution, and artifact assertions together."
    )]
    fn capture_cli_records_every_microphone_for_each_output() {
        use sotf_capture::signal_recorder::{
            FakeAudioConfig, FakeRecorder, NegotiatedInput, RecordedTake, SweepRecorder,
            VirtualWire,
        };

        struct TrackedRecorder {
            inner: FakeRecorder,
            starts: Vec<Vec<u16>>,
        }
        impl SweepRecorder for TrackedRecorder {
            fn start(
                &mut self,
                device: Option<&str>,
                channels: &[u16],
                rate: u32,
                seconds: f64,
            ) -> Result<NegotiatedInput, String> {
                self.starts.push(channels.to_vec());
                self.inner.start(device, channels, rate, seconds)
            }
            fn captured_samples(&self) -> usize {
                self.inner.captured_samples()
            }
            fn stop(&mut self) -> Result<RecordedTake, String> {
                self.inner.stop()
            }
        }

        // Also exercise equal list lengths: inputs are all microphones for
        // every output, rather than zipped output/input pairs.
        for microphone_count in [1, 3, 4] {
            let directory = tempfile::tempdir().unwrap();
            // Nonlinearity produces measurable sweep harmonics. Generic analysis
            // reports zero THD, so this checks the CLI actually selects ESS analysis.
            let wire = VirtualWire::new(FakeAudioConfig {
                output_channels: 3,
                input_channels: microphone_count,
                cubic_distortion: 4.0,
                ..FakeAudioConfig::default()
            });
            let microphones: Vec<u16> = (0..microphone_count as u16).collect();
            let microphone_list = microphones
                .iter()
                .map(u16::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let mut recorder = TrackedRecorder {
                inner: wire.recorder("cli regression"),
                starts: Vec::new(),
            };
            record_signal(
                &mut wire.playback("cli regression"),
                &mut recorder,
                "sweep".into(),
                2.0,
                8000,
                1,
                "0,1,2".into(),
                microphone_list,
                None,
                Some(directory.path().to_path_buf()),
                None,
                None,
                None,
                None,
                None,
                None,
                20.0,
                3000.0,
                Some(0.5),
                None,
                None,
                None,
                None,
                HashMap::new(),
                false,
            )
            .unwrap();
            // Wake probes use just the first mic; each sweep uses the whole list.
            if microphone_count > 1 {
                assert_eq!(
                    recorder
                        .starts
                        .iter()
                        .filter(|channels| **channels == microphones)
                        .count(),
                    3
                );
            }
            for output in 0..3 {
                for microphone in 0..microphone_count {
                    let stem = format!("sweep_send{output}_rec{microphone}_8000");
                    let wav = hound::WavReader::open(directory.path().join(format!("{stem}.wav")))
                        .unwrap();
                    assert_eq!(wav.spec().channels, 1);
                    assert_eq!(wav.spec().sample_rate, 8000);
                    let csv = std::fs::read_to_string(directory.path().join(format!("{stem}.csv")))
                        .unwrap();
                    let has_harmonics = csv.lines().skip(1).any(|line| {
                        let fields: Vec<_> = line.split(',').collect();
                        let frequency: f32 = fields[0].parse().unwrap();
                        let thd: f32 = fields[3].parse().unwrap();
                        (100.0..=500.0).contains(&frequency) && thd > 0.1
                    });
                    assert!(
                        has_harmonics,
                        "output {output} did not receive ESS analysis"
                    );
                }
            }
        }
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
                ir_microphone,
            } => {
                assert_eq!(raw, PathBuf::from("session/capture-raw.json"));
                assert_eq!(output, PathBuf::from("processed"));
                assert_eq!(selected_take_ids.len(), 2);
                assert_eq!(selected_take_ids[0], "take-r000-0123456789abcdef");
                assert_eq!(selected_take_ids[1], "take-r000-fedcba9876543210");
                assert_eq!(ir_microphone, None);
            }
            _ => panic!("expected plan-process command"),
        }
    }

    #[test]
    fn probe_command_parses_channels_levels_and_devices() {
        let cli = Cli::try_parse_from([
            "sotf-capture",
            "probe",
            "--hwaudio-send-to",
            "0,1",
            "--channel-names",
            "L,R",
            "--hwaudio-record-from",
            "0",
            "--probe-duration-ms",
            "500",
            "--signal-level-db",
            "-12",
            "--output-device",
            "dac",
            "--input-device",
            "mic",
            "--json",
        ])
        .unwrap();
        match cli.command {
            Command::Probe {
                hwaudio_send_to,
                channel_names,
                hwaudio_record_from,
                probe_duration_ms,
                signal_level_db,
                output_device,
                input_device,
                json,
                ..
            } => {
                assert_eq!(hwaudio_send_to, "0,1");
                assert_eq!(channel_names.as_deref(), Some("L,R"));
                assert_eq!(hwaudio_record_from, "0");
                assert_eq!(probe_duration_ms, 500.0);
                assert_eq!(signal_level_db, -12.0);
                assert_eq!(output_device.as_deref(), Some("dac"));
                assert_eq!(input_device.as_deref(), Some("mic"));
                assert!(json);
            }
            _ => panic!("expected probe command"),
        }
    }

    #[test]
    fn bass_anchor_command_parses_tone_and_loopback() {
        let cli = Cli::try_parse_from([
            "sotf-capture",
            "bass-anchor",
            "--hwaudio-send-to",
            "2",
            "--hwaudio-record-from",
            "0",
            "--loopback-from",
            "1",
            "--bass-freq-hz",
            "25",
            "--num-windows",
            "16",
        ])
        .unwrap();
        match cli.command {
            Command::BassAnchor {
                hwaudio_send_to,
                loopback_from,
                bass_freq_hz,
                num_windows,
                channel_names,
                ..
            } => {
                assert_eq!(hwaudio_send_to, "2");
                assert_eq!(loopback_from, Some(1));
                assert_eq!(bass_freq_hz, 25.0);
                assert_eq!(num_windows, 16);
                assert_eq!(channel_names, None);
            }
            _ => panic!("expected bass-anchor command"),
        }
    }

    #[test]
    fn aux_channel_resolution_enforces_single_input_and_named_outputs() {
        let (send, names, record) = resolve_aux_channels("0,1", "0", None).unwrap();
        assert_eq!(send, vec![0, 1]);
        assert_eq!(names, vec!["ch0".to_string(), "ch1".to_string()]);
        assert_eq!(record, 0);
        let (_, names, _) = resolve_aux_channels("0,1", "0", Some("L,R")).unwrap();
        assert_eq!(names, vec!["L".to_string(), "R".to_string()]);
        assert!(resolve_aux_channels("0,1", "0", Some("L")).is_err());
        assert!(resolve_aux_channels("0,1", "0", Some("L,")).is_err());
        assert!(resolve_aux_channels("0,1", "0,1", None).is_err());
        assert!(resolve_aux_channels("", "0", None).is_err());
    }

    #[test]
    fn plan_ctc_merge_parses_session_list_and_output() {
        let cli = Cli::try_parse_from([
            "sotf-capture",
            "plan-ctc-merge",
            "--session",
            "head-0",
            "--session",
            "head-1",
            "--output",
            "merged",
        ])
        .unwrap();
        match cli.command {
            Command::PlanCtcMerge { sessions, output } => {
                assert_eq!(
                    sessions,
                    vec![PathBuf::from("head-0"), PathBuf::from("head-1")]
                );
                assert_eq!(output, PathBuf::from("merged"));
            }
            _ => panic!("expected plan-ctc-merge command"),
        }
        assert!(
            Cli::try_parse_from(["sotf-capture", "plan-ctc-merge", "--output", "merged"]).is_err()
        );
    }

    #[test]
    fn plan_process_accepts_ir_microphone_selector() {
        let cli = Cli::try_parse_from([
            "sotf-capture",
            "plan-process",
            "--raw",
            "session/capture-raw.json",
            "--output",
            "processed",
            "--ir-microphone",
            "seat-left",
        ])
        .unwrap();
        match cli.command {
            Command::PlanProcess { ir_microphone, .. } => {
                assert_eq!(ir_microphone.as_deref(), Some("seat-left"));
            }
            _ => panic!("expected plan-process command"),
        }
    }

    #[test]
    fn plan_validation_checks_all_calibration_paths() {
        let directory = tempfile::tempdir().unwrap();
        let plan_directory = directory.path().join("plans");
        std::fs::create_dir(&plan_directory).unwrap();
        let calibration = directory.path().join("calibration.txt");
        std::fs::write(&calibration, "20 0\n20000 0\n").unwrap();
        let mut plan: serde_json::Value =
            serde_json::from_str(include_str!("../../examples/full-roomeq-session.json")).unwrap();
        for mic in plan["microphones"].as_array_mut().unwrap() {
            mic["calibration_file"] = "../calibration.txt".into();
        }
        // Absolute paths and paths relative to the plan both resolve successfully.
        plan["microphones"][1]["calibration_file"] = calibration.to_str().unwrap().into();
        let plan_path = plan_directory.join("session.json");
        std::fs::write(&plan_path, serde_json::to_vec(&plan).unwrap()).unwrap();
        plan_validate(&plan_path).unwrap();

        plan["microphones"][0]["calibration_file"] = "missing-first.txt".into();
        plan["microphones"][1]["calibration_file"] = "missing-second.txt".into();
        plan["microphones"][2]["calibration_file"] = ".".into();
        std::fs::write(&plan_path, serde_json::to_vec(&plan).unwrap()).unwrap();
        let error = plan_validate(&plan_path).unwrap_err();
        for (id, filename) in [
            ("main-seat", "missing-first.txt"),
            ("second-seat", "missing-second.txt"),
            ("ear-left", "."),
        ] {
            assert!(error.contains(id), "{error}");
            assert!(
                error.contains(&plan_directory.join(filename).display().to_string()),
                "{error}"
            );
        }
        assert!(error.contains("not a regular file"), "{error}");
        assert!(!error.contains("ear-right"), "{error}");
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
}
