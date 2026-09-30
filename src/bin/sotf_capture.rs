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
        /// Audio device name (default devices when omitted).
        #[arg(long)]
        device: Option<String>,
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
        println!("{}", serde_json::to_string_pretty(&devices).unwrap_or_default());
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
    use sotf_capture::recording_helpers::{capture_signal_params, summarize_take_quality};
    use sotf_capture::recording_helpers::take_verdict_text;
    use sotf_capture::signal_recorder::{
        CpalPlayback, DEFAULT_MLS_ORDER, SignalParams, SignalType, generate_output_filenames_stereo,
        generate_signal, parse_channel_list, prepare_measurement_signal, prepare_signal,
        record_and_analyze_with, validate_signal_params, write_temp_wav,
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
        SignalType::WhiteNoise | SignalType::PinkNoise | SignalType::MNoise => SignalParams::Noise {
            amp: validated_amp(amp, "--amp")?,
        },
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
            let compensation = math_audio_dsp::analysis::MicrophoneCompensation::from_file(
                Path::new(path),
            )?;
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
            device.as_deref(),
            device.as_deref(),
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
                    "send_channel": send_ch,
                    "record_channel": record_ch,
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
            "Recording source {}/{}: {} (all microphones)",
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

fn parse_mic_calibrations(values: &[String]) -> Result<HashMap<usize, String>, String> {
    let mut map = HashMap::new();
    for value in values {
        let (channel, path) = value
            .split_once('=')
            .ok_or_else(|| format!("--mic-calibration expects CH=PATH, got {value:?}"))?;
        let channel: usize = channel
            .parse()
            .map_err(|_| format!("invalid --mic-calibration channel: {channel:?}"))?;
        map.insert(channel, path.to_string());
    }
    Ok(map)
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Command::Devices { json } => list_audio_devices(json),
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
    }
}

fn main() {
    env_logger::init();
    if let Err(error) = run() {
        eprintln!("sotf-capture: {error}");
        std::process::exit(1);
    }
}
