#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "Synthetic fixtures use bounded sample counts and intentionally quantized audio; exact comparisons verify unchanged settings and samples."
)]

use super::*;
use crate::capture_session::protocol::prepare_capture_stimulus;
use crate::capture_session::record::RawCaptureStatus;
use crate::capture_session::{CaptureGeometry, CaptureSessionPlan, CaptureTimingReference};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

#[derive(Deserialize)]
struct Golden {
    sample_rate_hz: u32,
    offset_samples: [f64; 2],
    skew_ppm: [f64; 2],
    microphone_positions_m: [[f64; 3]; 2],
    timing_emitter_position_m: [f64; 3],
    sound_speed_m_s: f64,
}

fn fixture() -> (Golden, RawCaptureManifest, Vec<f32>, Vec<f32>) {
    let golden: Golden = serde_json::from_str(include_str!(
        "../../../tests/fixtures/capture-clock-golden.json"
    ))
    .unwrap();
    let mut plan: CaptureSessionPlan =
        serde_json::from_str(include_str!("../../../tests/fixtures/capture-session.json")).unwrap();
    plan.sample_rate_hz = golden.sample_rate_hz;
    plan.geometry = CaptureGeometry::Compact;
    plan.sweep.duration_secs = 0.05;
    for (mic, position) in plan
        .microphones
        .iter_mut()
        .zip(golden.microphone_positions_m)
    {
        mic.position_m = Some(position);
        mic.position_uncertainty_mm = Some(0.1);
    }
    plan.timing_reference = Some(CaptureTimingReference {
        output_channel: 0,
        position_m: golden.timing_emitter_position_m,
        position_uncertainty_mm: 0.1,
        sound_speed_m_s: golden.sound_speed_m_s,
        sound_speed_uncertainty_m_s: 0.1,
    });
    let session = plan.validate().unwrap();
    let stimulus = prepare_capture_stimulus(&session).unwrap();
    let manifest = RawCaptureManifest {
        localization_calibration: None,
        version: 1,
        plan: session.plan().clone(),
        stimulus: stimulus.layout,
        timing_reference_output_channel: Some(0),
        status: RawCaptureStatus::RawComplete,
        pending_processing: vec!["clock_correction".into()],
        calibrations: vec![],
        takes: vec![],
        loopback_files: vec![],
        error: None,
        probe_pass: None,
        bass_anchor_pass: None,
    };
    (golden, manifest, stimulus.samples, stimulus.timing_chirp)
}

// Independent first-order-hold sampling oracle. This intentionally does not use
// the production FIR resampler; the interpolation adds a small waveform error.
fn sample_clock(stimulus: &[f32], offset: f64, ppm: f64, propagation: f64) -> Vec<f32> {
    (0..stimulus.len() + offset.max(0.0).ceil() as usize + 4096)
        .map(|n| {
            let time = (n as f64 - offset) / (1.0 + ppm / 1e6) - propagation;
            if time < 0.0 || time >= (stimulus.len() - 1) as f64 {
                return 0.0;
            }
            let index = time.floor() as usize;
            let frac = (time - index as f64) as f32;
            stimulus[index] * (1.0 - frac) + stimulus[index + 1] * frac
        })
        .collect()
}

fn take(manifest: &RawCaptureManifest, index: usize, samples: usize) -> RawCaptureTake {
    RawCaptureTake {
        take_id: crate::capture_session::record::stable_take_id(
            "left",
            &manifest.plan.microphones[index].id,
            0,
        ),
        repeat_index: 0,
        device_id: format!("usb-{index}"),
        output_device_id: "dac".into(),
        source_id: "left".into(),
        microphone_id: manifest.plan.microphones[index].id.clone(),
        wav_file: format!("mic-{index}.wav"),
        samples,
        input_sample_format: "F32".into(),
        output_sample_format: "F32".into(),
        peak_amplitude: 0.1,
        clipped_samples: 0,
    }
}

#[test]
fn golden_two_microphone_clock_fit_preserves_physical_arrival_differences() {
    let (golden, manifest, stimulus, chirp) = fixture();
    for index in 0..2 {
        let propagation = (1.0 - golden.microphone_positions_m[index][0]) / golden.sound_speed_m_s
            * f64::from(golden.sample_rate_hz);
        let input = sample_clock(
            &stimulus,
            golden.offset_samples[index],
            golden.skew_ppm[index],
            propagation,
        );
        let result = correct_capture_clock(
            &manifest,
            &take(&manifest, index, input.len()),
            &input,
            &chirp,
        );
        let clock = result.provenance;
        assert_eq!(
            clock.basis,
            CaptureClockBasis::FixedAcousticReference,
            "{clock:?}"
        );
        assert!(
            (clock.skew_ppm.unwrap() - golden.skew_ppm[index]).abs() < 2.0,
            "{clock:?}"
        );
        assert!(
            (clock.offset_samples.unwrap() - golden.offset_samples[index]).abs() < 1.0,
            "{clock:?}"
        );
        let bound = clock.residual_uncertainty_us.unwrap();
        assert!(bound.is_finite() && bound > 0.0);
        let corrected = result.samples.unwrap();
        let measured = estimate_chirp_tdoa(
            &chirp,
            &corrected[..manifest.stimulus.sweep_offset],
            &TdoaConfig::default(),
        );
        let error_samples =
            (measured.offset_samples - manifest.stimulus.start_chirp_offset as f64 - propagation)
                .abs();
        assert!(
            error_samples / f64::from(golden.sample_rate_hz) * 1e6 < bound,
            "error {error_samples}, bound {bound}"
        );
        assert!(
            measured.offset_samples - manifest.stimulus.start_chirp_offset as f64 > 100.0,
            "physical propagation must not be silently removed"
        );
    }
}

#[test]
fn missing_end_chirp_keeps_the_take_magnitude_only() {
    let (golden, manifest, stimulus, chirp) = fixture();
    let mut input = sample_clock(&stimulus, golden.offset_samples[0], golden.skew_ppm[0], 0.0);
    let start =
        (manifest.stimulus.end_chirp_offset as f64 * 1.00008 + golden.offset_samples[0]) as usize;
    input[start.saturating_sub(32)..].fill(0.0);
    let result = correct_capture_clock(&manifest, &take(&manifest, 0, input.len()), &input, &chirp);
    assert!(result.samples.is_none());
    assert_eq!(result.provenance.correction_applied, "none");
    assert!(result.provenance.residual_uncertainty_us.is_none());
    assert!(
        result
            .provenance
            .magnitude_only_reason
            .unwrap()
            .contains("end timing chirp")
    );
}

#[test]
fn absent_survey_never_turns_arrival_alignment_into_coherent_evidence() {
    let (golden, mut manifest, stimulus, chirp) = fixture();
    manifest.plan.timing_reference = None;
    let input = sample_clock(
        &stimulus,
        golden.offset_samples[0],
        golden.skew_ppm[0],
        140.0,
    );
    let result = correct_capture_clock(&manifest, &take(&manifest, 0, input.len()), &input, &chirp);
    assert!(result.samples.is_some());
    assert_eq!(result.provenance.basis, CaptureClockBasis::ArrivalAligned);
    assert!(result.provenance.residual_uncertainty_us.is_none());
    assert!(result.provenance.magnitude_only_reason.is_some());
}

fn write_golden_directory() -> tempfile::TempDir {
    use crate::capture_session::record::CaptureCalibration;
    use crate::signal_recorder::write_wav_file;
    let root = tempfile::tempdir().unwrap();
    let (golden, mut manifest, stimulus, chirp) = fixture();
    manifest.plan.sources.truncate(1);
    write_wav_file(
        &root.path().join("stimulus.wav"),
        &stimulus,
        golden.sample_rate_hz,
        1,
    )
    .unwrap();
    write_wav_file(
        &root.path().join("timing-chirp.wav"),
        &chirp,
        golden.sample_rate_hz,
        1,
    )
    .unwrap();
    for index in 0..2 {
        let propagation = (1.0 - golden.microphone_positions_m[index][0]) / golden.sound_speed_m_s
            * f64::from(golden.sample_rate_hz);
        let mut input = sample_clock(
            &stimulus,
            golden.offset_samples[index],
            golden.skew_ppm[index],
            propagation,
        );
        if index == 1 {
            let end_start = ((manifest.stimulus.end_chirp_offset as f64 + propagation)
                * (1.0 + golden.skew_ppm[index] / 1e6)
                + golden.offset_samples[index]) as usize;
            input[end_start.saturating_sub(32)..].fill(0.0);
        }
        let take = take(&manifest, index, input.len());
        write_wav_file(
            &root.path().join(&take.wav_file),
            &input,
            golden.sample_rate_hz,
            1,
        )
        .unwrap();
        manifest.takes.push(take);
        let bytes = b"20 0\n1000 0\n20000 0\n";
        let file = format!("calibration-{index}.txt");
        std::fs::write(root.path().join(&file), bytes).unwrap();
        manifest.calibrations.push(CaptureCalibration {
            microphone_id: manifest.plan.microphones[index].id.clone(),
            file,
            sha256: crate::capture_session::calibration_sha256(bytes),
        });
    }
    std::fs::write(
        root.path().join("capture-raw.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    root
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "This end-to-end fixture keeps setup, execution, and artifact assertions together."
)]
fn saved_golden_session_round_trips_clock_provenance_and_magnitude_fallback() {
    let root = write_golden_directory();
    let output = root.path().join("processed");
    let report = io::process_capture_session(root.path(), &output).unwrap();
    assert_eq!(report.takes.len(), 2);
    assert_eq!(report.reflection_reports.len(), report.plan.sources.len());
    assert!(
        report
            .reflection_reports
            .iter()
            .all(|source| !source.issues.is_empty())
    );
    assert_eq!(report.takes[0].clock.correction_applied, "resampled");
    assert!(report.takes[0].clock.residual_uncertainty_us.is_some());
    assert_eq!(report.takes[1].clock.correction_applied, "none");
    assert!(report.takes[1].clock.residual_uncertainty_us.is_none());
    assert_eq!(
        report.recording_manifest.as_deref(),
        Some("recordings.json"),
        "{:?}",
        report.pending_processing
    );
    let handoff: autoeq::capture_handoff::CaptureHandoff = serde_json::from_slice(
        &std::fs::read(output.join(autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME)).unwrap(),
    )
    .unwrap();
    handoff.validate().unwrap();
    assert_eq!(
        handoff.completion,
        autoeq::capture_handoff::CaptureCompletion::Complete
    );
    assert_eq!(handoff.takes.len(), 2);
    for (index, take) in handoff.takes.iter().enumerate() {
        let retained = std::fs::read(output.join(&take.raw_audio_file)).unwrap();
        let original = std::fs::read(root.path().join(&report.takes[index].raw.wav_file)).unwrap();
        assert_eq!(retained, original);
        let identity = handoff
            .artifacts
            .iter()
            .find(|asset| asset.file == take.raw_audio_file)
            .unwrap();
        assert_eq!(identity.sha256.len(), 64);
        assert!(
            identity
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
        let decoded = identity
            .sha256
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let expected = Sha256::digest(&retained);
        let expected_bytes: &[u8] = expected.as_ref();
        assert_eq!(decoded.as_slice(), expected_bytes);
    }
    // Exercise the actual producer-to-consumer handoff.
    autoeq::roomeq::load_config(&output.join("recordings.json"), None).unwrap();
    let configuration: autoeq::roomeq::RoomConfig =
        serde_json::from_slice(&std::fs::read(output.join("recordings.json")).unwrap()).unwrap();
    let autoeq::roomeq::SpeakerConfig::Single(source) = &configuration.speakers["left"] else {
        panic!("expected captured source");
    };
    let autoeq::roomeq::MeasurementSource::Multiple(multiple) = source else {
        panic!("expected ordered microphone measurements");
    };
    let capture = multiple.provenance.capture.as_ref().unwrap();
    assert_eq!(capture.takes.len(), 2);
    assert_eq!(
        capture.reflection_report.as_ref().unwrap().source_id,
        "left"
    );
    assert_ne!(
        capture.reflection_report.as_ref().unwrap().issues,
        [] as [std::string::String; 0]
    );
    assert_eq!(
        capture.takes[0].microphone_id,
        report.plan.microphones[0].id
    );
    assert_eq!(
        capture.takes[0].calibration_id,
        report.calibrations[0].sha256
    );
    assert_eq!(
        capture.takes[1].correction_applied,
        autoeq::capture_provenance::CaptureCorrection::None
    );
    assert!(capture.takes[1].residual_uncertainty_us.is_none());
    assert!(source.provenance().timing_reference_id.is_none());
    assert_eq!(
        source.provenance().capture_kind,
        autoeq::ProvenanceCaptureKind::SpatialMagnitude
    );
    let mut incomplete = report.clone();
    incomplete.takes.pop();
    assert!(crate::capture_session::manifest::recording_configuration(&incomplete).is_err());
    let original = hound::WavReader::open(root.path().join(&report.takes[1].raw.wav_file))
        .unwrap()
        .into_samples::<f32>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let fallback = hound::WavReader::open(output.join(&report.takes[1].audio_file))
        .unwrap()
        .into_samples::<f32>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        original, fallback,
        "degraded timing must preserve usable raw magnitude samples"
    );
    let persisted: io::ClockProcessedManifest =
        serde_json::from_slice(&std::fs::read(output.join("capture-clock.json")).unwrap()).unwrap();
    assert_eq!(
        persisted.takes[0].clock.reference_id.as_deref(),
        Some("dac:0")
    );
    assert!(
        io::process_capture_session(root.path(), &output).is_err(),
        "never overwrite existing artifacts"
    );
}

#[test]
fn unknown_ir_microphone_is_refused_before_writing() {
    let root = write_golden_directory();
    let output = root.path().join("processed");
    let error = io::process_capture_session_with_options(
        root.path(),
        &output,
        &io::ProcessOptions {
            selected_take_ids: Vec::new(),
            ir_microphone: Some("seat-ghost".into()),
        },
    )
    .unwrap_err();
    assert!(error.contains("unknown IR microphone"), "{error}");
    assert!(!output.exists());
}

#[test]
fn ir_microphone_defaults_to_first_plan_microphone() {
    let root = write_golden_directory();
    let output = root.path().join("processed");
    let report = io::process_capture_session_with_options(
        root.path(),
        &output,
        &io::ProcessOptions::default(),
    )
    .unwrap();
    let expected = report.plan.microphones[0].id.clone();
    assert_eq!(report.ir_microphone.as_deref(), Some(expected.as_str()));
}

#[test]
fn auxiliary_passes_flow_into_projection_and_handoff() {
    use crate::capture_session::record::{CaptureBassAnchorPass, CaptureProbePass};
    use crate::signal_recorder::{
        BassAnchorChannelResult, BassAnchorResults, ProbeDelayChannelResult, ProbeDelayResults,
    };

    let root = write_golden_directory();
    let mut manifest: crate::capture_session::record::RawCaptureManifest =
        serde_json::from_slice(&std::fs::read(root.path().join("capture-raw.json")).unwrap())
            .unwrap();
    manifest.probe_pass = Some(CaptureProbePass {
        wav_file: "probe.wav".into(),
        microphone_id: manifest.plan.microphones[0].id.clone(),
        results: ProbeDelayResults {
            channels: vec![ProbeDelayChannelResult {
                channel_name: "left".into(),
                channel_index: 0,
                arrival_ms: 1.5,
                gain_db: -3.0,
                snr_db: 40.0,
            }],
            sample_rate: golden_sample_rate(&manifest),
            alignment_delays_ms: vec![0.5],
        },
    });
    manifest.bass_anchor_pass = Some(CaptureBassAnchorPass {
        wav_file: "bass-anchor.wav".into(),
        microphone_id: manifest.plan.microphones[0].id.clone(),
        loopback_channel: None,
        results: BassAnchorResults {
            channels: vec![BassAnchorChannelResult {
                channel_name: "left".into(),
                channel_index: 0,
                bass_anchor_phase_deg: 12.0,
                bass_anchor_magnitude: 0.5,
                bass_anchor_stability_deg: 2.0,
                bass_anchor_loopback_phase_deg: None,
                bass_anchor_coherence: None,
            }],
            sample_rate: golden_sample_rate(&manifest),
            bass_freq_hz: 30.0,
            bass_duration_s: 2.0,
        },
    });
    std::fs::write(
        root.path().join("capture-raw.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    for wav in ["probe.wav", "bass-anchor.wav"] {
        crate::signal_recorder::write_wav_file(
            &root.path().join(wav),
            &[0.1, -0.1, 0.05],
            golden_sample_rate(&manifest),
            1,
        )
        .unwrap();
    }
    let output = root.path().join("processed");
    let report = io::process_capture_session(root.path(), &output).unwrap();
    assert!(output.join("probe.wav").is_file());
    assert!(output.join("bass-anchor.wav").is_file());
    assert!(report.probe_pass.is_some());
    assert!(report.bass_anchor_pass.is_some());
    let configuration: autoeq::roomeq::RoomConfig =
        serde_json::from_slice(&std::fs::read(output.join("recordings.json")).unwrap()).unwrap();
    let recording = configuration.recording_config.as_ref().unwrap();
    assert_eq!(
        recording.probe_results.as_ref().unwrap().channels[0].arrival_ms,
        1.5
    );
    assert_eq!(recording.probe_wav_relative.as_deref(), Some("probe.wav"));
    assert_eq!(
        recording.bass_anchor_results.as_ref().unwrap().channels[0].bass_anchor_phase_deg,
        12.0
    );
    assert_eq!(recording.bass_probe_freq_hz, Some(30.0));
    let handoff: autoeq::capture_handoff::CaptureHandoff = serde_json::from_slice(
        &std::fs::read(output.join(autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME)).unwrap(),
    )
    .unwrap();
    handoff.validate().unwrap();
    for wav in ["probe.wav", "bass-anchor.wav"] {
        assert!(
            handoff.artifacts.iter().any(|asset| asset.file == wav),
            "handoff inventories {wav}"
        );
    }
    autoeq::roomeq::load_config(&output.join("recordings.json"), None).unwrap();
}

fn golden_sample_rate(manifest: &crate::capture_session::record::RawCaptureManifest) -> u32 {
    manifest.plan.sample_rate_hz
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "This end-to-end fixture keeps setup, execution, and artifact assertions together."
)]
fn ctc_assembly_crops_raw_takes_into_stereo_ears_loopback_and_reference() {
    use crate::capture_session::record::CaptureLoopbackTake;
    use crate::capture_session::{CaptureCtc, CaptureHeadPosition, CaptureLoopback};

    let root = write_golden_directory();
    let mut manifest: crate::capture_session::record::RawCaptureManifest =
        serde_json::from_slice(&std::fs::read(root.path().join("capture-raw.json")).unwrap())
            .unwrap();
    manifest.plan.loopback = Some(CaptureLoopback {
        device: manifest.plan.microphones[0].device.clone(),
        input_channel: 7,
    });
    manifest.plan.ctc = Some(CaptureCtc {
        left_ear: manifest.plan.microphones[0].id.clone(),
        right_ear: manifest.plan.microphones[1].id.clone(),
        head_position: CaptureHeadPosition {
            id: "head-0".into(),
            x: 1.0,
            y: 0.0,
            z: 1.2,
            yaw_deg: 0.0,
        },
    });
    // Independent sampling oracle: one hardware timeline, a three-second
    // startup offset, nonzero drift, and different physical ear delays.
    for (index, mic) in manifest.plan.microphones.iter_mut().enumerate() {
        mic.device = "shared-input".into();
        mic.input_channel = index as u16;
    }
    manifest.plan.loopback.as_mut().unwrap().device = "shared-input".into();
    let prepared = prepare_capture_stimulus(&manifest.plan.clone().validate().unwrap()).unwrap();
    let offset = 144_000.0;
    let ppm = 60.0;
    for (index, delay) in [96.0, 120.0].into_iter().enumerate() {
        let samples = sample_clock(&prepared.samples, offset, ppm, delay);
        let take = &mut manifest.takes[index];
        take.device_id = "shared-input".into();
        take.samples = samples.len();
        crate::signal_recorder::write_wav_file(
            &root.path().join(&take.wav_file),
            &samples,
            manifest.plan.sample_rate_hz,
            1,
        )
        .unwrap();
    }
    let electrical = sample_clock(&prepared.samples, offset, ppm, 0.0);
    manifest.loopback_files.push(CaptureLoopbackTake {
        repeat_index: 0,
        source_id: manifest.plan.sources[0].id.clone(),
        wav_file: "loopback.wav".into(),
        samples: electrical.len(),
        device_id: "shared-input".into(),
        input_sample_format: manifest.takes[0].input_sample_format.clone(),
        peak_amplitude: 0.1,
        clipped_samples: 0,
    });
    crate::signal_recorder::write_wav_file(
        &root.path().join("loopback.wav"),
        &electrical,
        manifest.plan.sample_rate_hz,
        1,
    )
    .unwrap();
    std::fs::write(
        root.path().join("capture-raw.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let output = root.path().join("processed");
    let report = io::process_capture_session(root.path(), &output).unwrap();
    assert_eq!(
        report.ctc_measurements.as_deref(),
        Some(crate::capture_session::ctc::CTC_FRAGMENT_FILE)
    );
    assert_eq!(report.ctc_files.len(), 4);
    let window = report.stimulus.end_chirp_offset - report.stimulus.sweep_offset;
    let fragment: crate::capture_session::ctc::CtcFragment = serde_json::from_slice(
        &std::fs::read(output.join(crate::capture_session::ctc::CTC_FRAGMENT_FILE)).unwrap(),
    )
    .unwrap();
    assert_eq!(fragment.measurements.speakers, vec!["left".to_string()]);
    assert_eq!(
        fragment.measurements.mics,
        vec!["left_ear".to_string(), "right_ear".to_string()]
    );
    assert_eq!(fragment.measurements.files.len(), 1);
    assert_eq!(fragment.measurements.files[0].head_position, "head-0");
    let entry = &fragment.measurements.files[0];
    let ears = entry.raw_sweep.as_deref().unwrap();
    let loopback = entry.loopback.as_deref().unwrap();
    let mut ears_reader = hound::WavReader::open(output.join(ears)).unwrap();
    assert_eq!(ears_reader.spec().channels, 2);
    assert_eq!(ears_reader.duration() as usize, window);
    let loopback_reader = hound::WavReader::open(output.join(loopback)).unwrap();
    assert_eq!(loopback_reader.spec().channels, 1);
    assert_eq!(loopback_reader.duration() as usize, window);
    let reference_reader = hound::WavReader::open(output.join(&fragment.reference_sweep)).unwrap();
    assert_eq!(
        reference_reader.duration() as usize,
        report.stimulus.sweep_samples
    );
    // Recover independently specified physical delays against the electrical
    // sweep, rather than asserting equality with implementation crop indices.
    let reference: Vec<f32> = hound::WavReader::open(output.join(&fragment.reference_sweep))
        .unwrap()
        .samples::<f32>()
        .collect::<Result<_, _>>()
        .unwrap();
    let stereo: Vec<f32> = ears_reader
        .samples::<f32>()
        .collect::<Result<_, _>>()
        .unwrap();
    let mut recovered = Vec::new();
    for (channel, expected) in [96.0, 120.0].into_iter().enumerate() {
        let ear: Vec<f32> = stereo.iter().skip(channel).step_by(2).copied().collect();
        let lag = math_audio_dsp::analysis::estimate_lag_with_confidence(&reference, &ear).unwrap();
        assert!((lag.lag_samples as f64 - expected).abs() < 2.0, "{lag:?}");
        recovered.push(lag.lag_samples as f64);
    }
    assert!((recovered[1] - recovered[0] - 24.0).abs() < 2.0);
    let handoff: autoeq::capture_handoff::CaptureHandoff = serde_json::from_slice(
        &std::fs::read(output.join(autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME)).unwrap(),
    )
    .unwrap();
    handoff.validate().unwrap();
    for file in &report.ctc_files {
        assert!(
            handoff.artifacts.iter().any(|asset| &asset.file == file),
            "handoff inventories {file}"
        );
    }
    // Matching plan selectors cannot excuse independently recorded clocks.
    manifest.takes[1].device_id = "other-clock".into();
    std::fs::write(
        root.path().join("capture-raw.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let rejected = root.path().join("independent");
    let error = io::process_capture_session(root.path(), &rejected).unwrap_err();
    assert!(error.contains("hardware input timeline"), "{error}");
    assert!(
        !rejected
            .join(crate::capture_session::ctc::CTC_FRAGMENT_FILE)
            .exists()
    );
    assert!(
        !rejected
            .join(autoeq::capture_handoff::CAPTURE_HANDOFF_FILENAME)
            .exists()
    );

    manifest.takes[1].device_id = "shared-input".into();
    std::fs::write(
        root.path().join("capture-raw.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    crate::signal_recorder::write_wav_file(
        &root.path().join("loopback.wav"),
        &vec![0.0; electrical.len()],
        manifest.plan.sample_rate_hz,
        1,
    )
    .unwrap();
    let rejected = root.path().join("missing-markers");
    let error = io::process_capture_session(root.path(), &rejected).unwrap_err();
    assert!(error.contains("loopback markers"), "{error}");
    assert!(
        !rejected
            .join(crate::capture_session::ctc::CTC_FRAGMENT_FILE)
            .exists()
    );
    let mut markers_only = prepared.samples.clone();
    markers_only[manifest.stimulus.sweep_offset
        ..manifest.stimulus.sweep_offset + manifest.stimulus.sweep_samples]
        .fill(0.0);
    let markers_only = sample_clock(&markers_only, offset, ppm, 0.0);
    crate::signal_recorder::write_wav_file(
        &root.path().join("loopback.wav"),
        &markers_only,
        manifest.plan.sample_rate_hz,
        1,
    )
    .unwrap();
    let rejected = root.path().join("markers-without-sweep");
    let error = io::process_capture_session(root.path(), &rejected).unwrap_err();
    assert!(error.contains("loopback sweep"), "{error}");
    assert!(
        !rejected
            .join(crate::capture_session::ctc::CTC_FRAGMENT_FILE)
            .exists()
    );
}

#[test]
fn complete_journal_without_declared_auxiliary_results_is_rejected_before_publication() {
    let root = write_golden_directory();
    let path = root.path().join("capture-raw.json");
    let mut manifest: RawCaptureManifest =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest.plan.probe = Some(crate::capture_session::CaptureProbe {
        probe_duration_ms: 100.0,
        silence_duration_ms: 100.0,
        signal_level_db: -20.0,
    });
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let output = root.path().join("processed");
    assert!(
        io::process_capture_session(root.path(), &output)
            .unwrap_err()
            .contains("missing declared auxiliary")
    );
    assert!(!output.exists());
}

#[test]
fn process_reports_mixed_timing_identities_without_naming_a_winner() {
    let root = write_golden_directory();
    let output = root.path().join("processed");
    let report = io::process_capture_session(root.path(), &output).unwrap();
    assert_eq!(report.timing_reference_output_channel, Some(0));
    let summary = io::timing_reference_summary(&report);
    assert!(summary.contains("no shared timing reference"), "{summary}");
    assert!(summary.contains("1 of 2 takes"), "{summary}");
    assert!(summary.contains("output channel 0"), "{summary}");
    assert!(summary.contains("forbid coherent use"), "{summary}");
}

#[test]
fn process_reports_fully_shared_timing_identity() {
    let root = write_golden_directory();
    let output = root.path().join("processed");
    let mut report = io::process_capture_session(root.path(), &output).unwrap();
    for take in &mut report.takes {
        take.clock.reference_id = Some("dac:0".into());
    }
    assert_eq!(
        io::timing_reference_summary(&report),
        "timing reference 'dac:0' shared by 2/2 takes on output channel 0"
    );
    report.timing_reference_output_channel = None;
    let legacy = io::timing_reference_summary(&report);
    assert!(legacy.contains("unknown route (legacy report)"), "{legacy}");
    assert!(legacy.contains("shared by 2/2 takes"), "{legacy}");
}

#[test]
fn canonical_phase_requires_complete_quality_and_retains_frequency_gate() {
    use crate::capture_session::analysis::{CaptureFrequencySnr, CapturePhaseArtifacts};
    let root = write_golden_directory();
    let output = root.path().join("processed");
    let mut report = io::process_capture_session(root.path(), &output).unwrap();
    let clock = report.takes[0].clock.clone();
    for (index, take) in report.takes.iter_mut().enumerate() {
        take.clock = clock.clone();
        take.clock.device_id = take.raw.device_id.clone();
        take.clock.residual_uncertainty_us = Some(20.0);
        take.clock.magnitude_only_reason = None;
        let analysis = take.analysis.as_mut().unwrap();
        analysis.issues.clear();
        analysis.clipped_samples = 0;
        analysis.broadband_snr_db = Some(40.0);
        analysis.frequency_snr = vec![CaptureFrequencySnr {
            low_hz: 20.0,
            high_hz: 20000.0,
            snr_db: Some(40.0),
        }];
        analysis.common_reference = Some(CapturePhaseArtifacts {
            response_file: format!("take-{index:03}-phase.csv"),
            impulse_file: format!("take-{index:03}-ir.json"),
            impulse_csv_file: None,
            sti: None,
            coherence_file: None,
            timing_limit_hz: 1250.0,
            low_hz: 20.0,
            high_hz: 20000.0,
        });
    }
    let config = crate::capture_session::manifest::recording_configuration(&report).unwrap();
    let autoeq::roomeq::SpeakerConfig::Single(source) = &config.speakers["left"] else {
        panic!()
    };
    assert_eq!(
        source.provenance().capture_kind,
        autoeq::ProvenanceCaptureKind::StationaryIr
    );
    let provenance = source.provenance();
    let capture = provenance.capture.unwrap();
    assert!(capture.coherent_reference_at_frequency(2, 500.0).is_ok());
    assert!(capture.coherent_reference_at_frequency(2, 20000.0).is_err());
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("take-000-phase.csv"));
    report.takes[1].analysis.as_mut().unwrap().common_reference = None;
    let magnitude_only =
        crate::capture_session::manifest::recording_configuration(&report).unwrap();
    let autoeq::roomeq::SpeakerConfig::Single(source) = &magnitude_only.speakers["left"] else {
        panic!("captured source");
    };
    let provenance = source.provenance();
    assert_eq!(
        provenance.capture_kind,
        autoeq::ProvenanceCaptureKind::SpatialMagnitude
    );
    assert!(
        provenance
            .capture
            .unwrap()
            .takes
            .iter()
            .all(|take| take.quality_passed),
        "missing shared phase must not erase separately accepted magnitude quality"
    );
    report.takes[1].analysis.as_mut().unwrap().frequency_snr[0].snr_db = None;
    let fallback = crate::capture_session::manifest::recording_configuration(&report).unwrap();
    let autoeq::roomeq::SpeakerConfig::Single(source) = &fallback.speakers["left"] else {
        panic!()
    };
    assert_eq!(
        source.provenance().capture_kind,
        autoeq::ProvenanceCaptureKind::SpatialMagnitude
    );
    assert!(
        !serde_json::to_string(&fallback)
            .unwrap()
            .contains("phase.csv")
    );
}

#[test]
fn legacy_clock_report_without_raw_status_requires_the_complete_single_repeat_matrix() {
    let root = write_golden_directory();
    let output = root.path().join("legacy-processed");
    let report = io::process_capture_session(root.path(), &output).unwrap();
    let mut value = serde_json::to_value(report).unwrap();
    let report = value.as_object_mut().unwrap();
    report.remove("raw_status");
    report.remove("selected_take_ids");
    report.remove("parent_inventory_file");
    value["plan"]
        .as_object_mut()
        .unwrap()
        .remove("repeat_count");

    let legacy: io::ClockProcessedManifest = serde_json::from_value(value).unwrap();
    let configuration = crate::capture_session::manifest::recording_configuration(&legacy)
        .expect("legacy reports retain complete single-repeat compatibility");
    assert_eq!(configuration.speakers.len(), legacy.plan.sources.len());

    let mut incomplete = legacy.clone();
    incomplete.takes.pop();
    assert!(
        crate::capture_session::manifest::recording_configuration(&incomplete).is_err(),
        "legacy compatibility must still require every source/microphone pair"
    );

    for status in [RawCaptureStatus::Cancelled, RawCaptureStatus::Failed] {
        let mut explicitly_incomplete = legacy.clone();
        explicitly_incomplete.raw_status = Some(status);
        assert!(
            crate::capture_session::manifest::recording_configuration(&explicitly_incomplete)
                .is_err(),
            "explicit {status:?} status must not use the legacy projection"
        );
    }
}

#[test]
fn altered_calibration_snapshot_is_rejected_before_output_creation() {
    let root = write_golden_directory();
    std::fs::write(root.path().join("calibration-1.txt"), "changed").unwrap();
    let output = root.path().join("processed");
    let error = io::process_capture_session(root.path(), &output).unwrap_err();
    assert!(error.contains("has changed"), "{error}");
    assert!(!output.exists());
}

#[test]
fn raw_journal_cannot_read_artifacts_outside_its_directory() {
    let root = write_golden_directory();
    let path = root.path().join("capture-raw.json");
    let mut manifest: RawCaptureManifest =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest.takes[0].wav_file = "../other.wav".into();
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(
        io::process_capture_session(root.path(), &root.path().join("processed"))
            .unwrap_err()
            .contains("relative filenames")
    );
}

#[test]
fn selection_rejects_duplicate_unknown_and_incomplete_take_ids_before_writing() {
    let root = write_golden_directory();
    let raw: RawCaptureManifest =
        serde_json::from_slice(&std::fs::read(root.path().join("capture-raw.json")).unwrap())
            .unwrap();
    let first = raw.takes[0].take_id.clone();
    let second = raw.takes[1].take_id.clone();
    let cases = [
        (vec![first.clone(), first], "selected take ID is duplicated"),
        (
            vec![second.clone(), "unknown-take".into()],
            "selected take ID is unknown",
        ),
        (
            vec![second],
            "exactly one completed take per source/microphone pair",
        ),
    ];
    for (index, (selected, expected)) in cases.into_iter().enumerate() {
        let output = root.path().join(format!("invalid-selection-{index}"));
        let error = io::process_capture_session_with_selection(root.path(), &output, &selected)
            .unwrap_err();
        assert!(error.contains(expected), "{error}");
        assert!(!output.exists());
    }
}

#[test]
fn raw_take_repeat_index_must_belong_to_the_plan() {
    let root = write_golden_directory();
    let path = root.path().join("capture-raw.json");
    let mut manifest: RawCaptureManifest =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest.takes[0].repeat_index = manifest.plan.repeat_count;
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let output = root.path().join("out-of-domain-repeat");
    let error = io::process_capture_session(root.path(), &output).unwrap_err();
    assert!(
        error.contains("unknown, duplicate, or unidentified"),
        "{error}"
    );
    assert!(!output.exists());
}

fn write_response_csv(dir: &std::path::Path, name: &str, freqs: &[f64]) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut text = String::from("frequency_hz,spl_db,phase_deg\n");
    for frequency in freqs {
        writeln!(text, "{frequency},80.0,0.0").unwrap();
    }
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn response_grid_csv_reads_first_column_and_skips_header() {
    let root = tempfile::tempdir().unwrap();
    let path = write_response_csv(root.path(), "take-000-phase.csv", &[20.0, 30.0, 40.0]);
    assert_eq!(io::response_grid_csv(&path), Some(vec![20.0, 30.0, 40.0]));
}

#[test]
fn response_grid_csv_skips_blank_lines_and_rejects_bad_grids() {
    let root = tempfile::tempdir().unwrap();
    let bad = root.path().join("bad.csv");
    std::fs::write(&bad, "frequency_hz,spl_db,phase_deg\n20,80,0\nNaN,80,0\n").unwrap();
    assert_eq!(io::response_grid_csv(&bad), None);
    let empty = root.path().join("empty.csv");
    std::fs::write(&empty, "frequency_hz,spl_db,phase_deg\n").unwrap();
    assert_eq!(io::response_grid_csv(&empty), None);
    assert_eq!(
        io::response_grid_csv(&root.path().join("missing.csv")),
        None
    );
}

#[test]
fn first_grid_mismatch_accepts_matching_single_and_empty_sets() {
    assert_eq!(io::first_grid_mismatch(&[]), None);
    assert_eq!(
        io::first_grid_mismatch(&[("take-000", vec![20.0, 30.0])]),
        None
    );
    assert_eq!(
        io::first_grid_mismatch(&[
            ("take-000", vec![20.0, 30.0]),
            ("take-001", vec![20.0, 30.0]),
        ]),
        None
    );
}

#[test]
fn first_grid_mismatch_reports_length_and_value_divergence() {
    let lengths =
        io::first_grid_mismatch(&[("take-000", vec![20.0, 30.0]), ("take-001", vec![20.0])])
            .expect("length mismatch must report");
    assert!(
        lengths.contains("take-000") && lengths.contains("take-001"),
        "{lengths}"
    );
    assert!(lengths.contains("2 vs 1"), "{lengths}");
    let values = io::first_grid_mismatch(&[
        ("take-000", vec![20.0, 30.0]),
        ("take-001", vec![20.0, 31.0]),
    ])
    .expect("value mismatch must report");
    assert!(values.contains("bin 1"), "{values}");
}

#[test]
fn clock_skew_summary_reports_per_take_fit_and_real_fallback() {
    let root = write_golden_directory();
    let output = root.path().join("processed");
    let report = io::process_capture_session(root.path(), &output).unwrap();
    let lines = io::clock_skew_lines(&report);
    assert_eq!(lines.len(), 2);
    // Take 0 resampled near the golden +80 ppm; take 1 fell back.
    assert!(
        lines[0].contains(&report.takes[0].raw.take_id) && lines[0].contains("resampled"),
        "{}",
        lines[0]
    );
    let skew = report.takes[0].clock.skew_ppm.unwrap();
    assert!(
        lines[0].contains(&format!("{skew:+.1}")),
        "line must print the fitted skew: {}",
        lines[0]
    );
    assert!(
        lines[1].contains("magnitude-only"),
        "fallback take must show its reason: {}",
        lines[1]
    );
    // Both real takes carry magnitude-only reasons (take 0 resampled but
    // its uncertainty exceeds the coherent gate); the quiet +80 ppm fit
    // itself raises no severe/wander warning.
    let warnings = io::clock_skew_warnings(&report);
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(
        warnings
            .iter()
            .all(|warning| warning.contains("magnitude-only")),
        "{warnings:?}"
    );
    assert!(
        warnings[1].contains(&report.takes[1].raw.take_id),
        "{}",
        warnings[1]
    );
}

#[test]
fn clock_skew_warnings_flag_severe_skew_and_wander() {
    let root = write_golden_directory();
    let output = root.path().join("processed");
    let mut report = io::process_capture_session(root.path(), &output).unwrap();
    report.takes[0].clock.skew_ppm = Some(150.0);
    report.takes[0].clock.magnitude_only_reason = None;
    report.takes[1].clock.skew_ppm = Some(-120.0);
    report.takes[1].clock.magnitude_only_reason = None;
    let warnings = io::clock_skew_warnings(&report);
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("+150.0") && warning.contains("exceeds 100 ppm")),
        "{warnings:?}"
    );
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("wanders 270 ppm")),
        "{warnings:?}"
    );
    // Small stable skews on every take stay quiet.
    report.takes[0].clock.skew_ppm = Some(40.0);
    report.takes[1].clock.skew_ppm = Some(-30.0);
    assert_eq!(
        io::clock_skew_warnings(&report),
        [] as [std::string::String; 0]
    );
}

#[test]
fn clock_skew_lines_mark_unknown_and_empty_runs() {
    let root = write_golden_directory();
    let output = root.path().join("processed");
    let mut report = io::process_capture_session(root.path(), &output).unwrap();
    report.takes[0].clock.skew_ppm = None;
    report.takes[0].clock.magnitude_only_reason = None;
    let lines = io::clock_skew_lines(&report);
    assert!(lines[0].contains("skew unknown"), "{}", lines[0]);
    assert!(
        io::clock_skew_warnings(&report)
            .iter()
            .all(|warning| !warning.contains("exceeds"))
    );
    report.takes.clear();
    assert_eq!(
        io::clock_skew_lines(&report),
        [] as [std::string::String; 0]
    );
    assert_eq!(
        io::clock_skew_warnings(&report),
        [] as [std::string::String; 0]
    );
}
