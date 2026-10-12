//! Hardware-free capture tests over [`VirtualWire`](super::fake::VirtualWire).

// Rust guideline compliant 2026-02-21

use super::fake::{FakeAudioConfig, VirtualWire};
use super::playback::SweepPlayback;
use super::record::{record_and_analyze_with_multi_recorder, record_and_analyze_with_recorder};
use super::recorder::SweepRecorder;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SAMPLE_RATE: u32 = 48_000;

/// Drift-estimator-friendly sweep (see `test_clock_drift_estimate_and_correct_roundtrip`).
fn sweep_2s() -> Vec<f32> {
    math_audio_dsp::signals::gen_log_sweep(100.0, 20_000.0, 0.5, SAMPLE_RATE, 2.0)
}

fn mean_and_variation(spl_db: &[f32]) -> (f32, f32) {
    let mean = spl_db.iter().sum::<f32>() / spl_db.len() as f32;
    let min = spl_db.iter().copied().fold(f32::INFINITY, f32::min);
    let max = spl_db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (mean, max - min)
}

fn band_spl(frequencies: &[f32], spl_db: &[f32], low_hz: f32, high_hz: f32) -> Vec<f32> {
    frequencies
        .iter()
        .zip(spl_db.iter())
        .filter(|(freq, _)| low_hz <= **freq && **freq <= high_hz)
        .map(|(_, spl)| *spl)
        .collect()
}

#[test]
fn fake_loopback_renders_stimulus_plus_silent_tail() {
    let wire = VirtualWire::default();
    let mut playback = wire.playback("fake");
    let mut recorder = wire.recorder("fake");

    let stimulus = vec![0.25_f32; 4_800];
    let temp_wav = super::write::write_temp_wav(&stimulus, SAMPLE_RATE, 1).unwrap();
    playback
        .start(temp_wav.path(), None, 0, SAMPLE_RATE)
        .unwrap();
    assert!(playback.is_finished());
    assert_eq!(wire.played_stimulus().unwrap().len(), 4_800);

    let negotiated = recorder.start(None, &[0], SAMPLE_RATE, 0.1).unwrap();
    assert_eq!(negotiated.sample_rate, SAMPLE_RATE);
    assert_eq!(negotiated.hardware_channels, 2);
    let take = recorder.stop().unwrap();
    assert_eq!(take.analysis_sample_rate, SAMPLE_RATE);
    assert_eq!(take.dropped_samples, 0);
    assert_eq!(take.recorded_per_channel.len(), 1);

    // Default 1 s tail of silence follows the stimulus bit-identically.
    let expected_len = 4_800 + SAMPLE_RATE as usize;
    assert_eq!(take.recorded_per_channel[0].len(), expected_len);
    assert!(
        take.recorded_per_channel[0][..4_800]
            .iter()
            .all(|s| (*s - 0.25).abs() < 1e-6)
    );
    assert!(
        take.recorded_per_channel[0][4_800..]
            .iter()
            .all(|s| *s == 0.0)
    );
}

#[test]
fn fake_recorder_rejects_unknown_and_missing_devices() {
    let wire = VirtualWire::default();
    let mut recorder = wire.recorder("record_and_analyze");

    let error = recorder
        .start(Some("no such mic"), &[0], SAMPLE_RATE, 1.0)
        .unwrap_err();
    assert!(
        error.contains("Input device not usable")
            && error.contains("Audio device 'no such mic' not found"),
        "{error}"
    );

    let config = FakeAudioConfig {
        has_default_input: false,
        ..FakeAudioConfig::default()
    };
    let mut recorder = VirtualWire::new(config).recorder("record_and_analyze");
    let error = recorder.start(None, &[0], SAMPLE_RATE, 1.0).unwrap_err();
    assert!(error.contains("No default input device"), "{error}");
}

#[test]
fn fake_backends_validate_channels() {
    let wire = VirtualWire::default();
    let mut recorder = wire.recorder("record_and_analyze");
    let error = recorder.start(None, &[7], SAMPLE_RATE, 1.0).unwrap_err();
    assert!(
        error.contains("Input channel 7 exceeds hardware channel count 2"),
        "{error}"
    );
    let error = recorder.start(None, &[], SAMPLE_RATE, 1.0).unwrap_err();
    assert!(error.contains("No input channels requested"), "{error}");

    let stimulus = vec![0.1_f32; 480];
    let temp_wav = super::write::write_temp_wav(&stimulus, SAMPLE_RATE, 1).unwrap();
    let mut playback = wire.playback("record_and_analyze");
    let error = playback
        .start(temp_wav.path(), None, 9, SAMPLE_RATE)
        .unwrap_err();
    assert!(
        error.contains("Output channel 9 exceeds hardware channel count 2"),
        "{error}"
    );
    let error = playback
        .start(temp_wav.path(), Some("no such output"), 0, SAMPLE_RATE)
        .unwrap_err();
    assert!(
        error.contains("Audio device 'no such output' not found"),
        "{error}"
    );
}

#[test]
fn fake_stop_without_start_is_an_error() {
    let wire = VirtualWire::default();
    let mut recorder = wire.recorder("record_and_analyze");
    let error = recorder.stop().unwrap_err();
    assert!(
        error.contains("stopped without a matching start"),
        "{error}"
    );
}

#[test]
fn fake_failure_injection_fails_start_and_stop() {
    let wire = VirtualWire::new(FakeAudioConfig {
        fail_input_start: Some("boom in".to_string()),
        fail_input_stop: Some("boom stop".to_string()),
        fail_output_start: Some("boom out".to_string()),
        ..FakeAudioConfig::default()
    });
    let mut playback = wire.playback("fake");
    let mut recorder = wire.recorder("fake");

    let stimulus = vec![0.1_f32; 480];
    let temp_wav = super::write::write_temp_wav(&stimulus, SAMPLE_RATE, 1).unwrap();
    assert_eq!(
        playback
            .start(temp_wav.path(), None, 0, SAMPLE_RATE)
            .unwrap_err(),
        "boom out"
    );
    assert_eq!(
        recorder.start(None, &[0], SAMPLE_RATE, 1.0).unwrap_err(),
        "boom in"
    );

    // Stop injection fires after a successful start.
    let wire = VirtualWire::new(FakeAudioConfig {
        fail_input_stop: Some("boom stop".to_string()),
        ..FakeAudioConfig::default()
    });
    let mut recorder = wire.recorder("fake");
    recorder.start(None, &[0], SAMPLE_RATE, 1.0).unwrap();
    assert_eq!(recorder.stop().unwrap_err(), "boom stop");
}

#[test]
fn fake_rate_negotiation_resamples_the_capture() {
    let wire = VirtualWire::new(FakeAudioConfig {
        negotiated_input_rate: Some(44_100),
        tail_secs: 0.0,
        ..FakeAudioConfig::default()
    });
    let mut playback = wire.playback("fake");
    let mut recorder = wire.recorder("fake");

    let stimulus = vec![0.25_f32; SAMPLE_RATE as usize];
    let temp_wav = super::write::write_temp_wav(&stimulus, SAMPLE_RATE, 1).unwrap();
    playback
        .start(temp_wav.path(), None, 0, SAMPLE_RATE)
        .unwrap();
    let negotiated = recorder.start(None, &[0], SAMPLE_RATE, 1.0).unwrap();
    assert_eq!(negotiated.sample_rate, 44_100);
    let take = recorder.stop().unwrap();
    assert_eq!(take.analysis_sample_rate, 44_100);
    assert_eq!(take.recorded_per_channel[0].len(), 44_100);
}

#[test]
fn fake_drift_stretches_the_capture() {
    // 1000 ppm over one second adds ~48 frames at 48 kHz.
    let wire = VirtualWire::new(FakeAudioConfig {
        drift_ppm: 1_000.0,
        tail_secs: 0.0,
        ..FakeAudioConfig::default()
    });
    let mut playback = wire.playback("fake");
    let mut recorder = wire.recorder("fake");

    let stimulus = vec![0.25_f32; SAMPLE_RATE as usize];
    let temp_wav = super::write::write_temp_wav(&stimulus, SAMPLE_RATE, 1).unwrap();
    playback
        .start(temp_wav.path(), None, 0, SAMPLE_RATE)
        .unwrap();
    recorder.start(None, &[0], SAMPLE_RATE, 1.0).unwrap();
    let take = recorder.stop().unwrap();
    assert_eq!(take.recorded_per_channel[0].len(), 48_048);
}

#[test]
fn fake_noise_is_deterministic_and_per_channel() {
    let config = FakeAudioConfig {
        noise_rms: 0.01,
        // Silent stimulus renders through the quiet window.
        quiet_window_noise_rms: 0.01,
        seed: 1234,
        tail_secs: 0.0,
        ..FakeAudioConfig::default()
    };
    let render = |channels: &[u16]| {
        let wire = VirtualWire::new(config.clone());
        let mut playback = wire.playback("fake");
        let mut recorder = wire.recorder("fake");
        let stimulus = vec![0.0_f32; 4_800];
        let temp_wav = super::write::write_temp_wav(&stimulus, SAMPLE_RATE, 1).unwrap();
        playback
            .start(temp_wav.path(), None, 0, SAMPLE_RATE)
            .unwrap();
        recorder.start(None, channels, SAMPLE_RATE, 0.1).unwrap();
        recorder.stop().unwrap().recorded_per_channel
    };

    let first = render(&[0, 1]);
    let second = render(&[0, 1]);
    assert_eq!(first, second);
    // Same seed, different channel index: independent noise streams.
    assert_ne!(first[0], first[1]);
    assert!(first[0].iter().any(|s| s.abs() > 0.005));
}

#[test]
fn fake_noise_uses_quiet_level_on_silent_samples() {
    // The two-knob noise model: silent samples render at
    // quiet_window_noise_rms, the rest at noise_rms.
    let config = FakeAudioConfig {
        noise_rms: 0.02,
        quiet_window_noise_rms: 0.001,
        seed: 1234,
        tail_secs: 0.0,
        ..FakeAudioConfig::default()
    };
    let render = |stimulus: Vec<f32>| {
        let wire = VirtualWire::new(config.clone());
        let mut playback = wire.playback("fake");
        let mut recorder = wire.recorder("fake");
        let temp_wav = super::write::write_temp_wav(&stimulus, SAMPLE_RATE, 1).unwrap();
        playback
            .start(temp_wav.path(), None, 0, SAMPLE_RATE)
            .unwrap();
        recorder.start(None, &[0], SAMPLE_RATE, 0.1).unwrap();
        recorder.stop().unwrap().recorded_per_channel
    };

    let loud = render(vec![0.25_f32; 4_800]);
    assert!(
        loud[0].iter().any(|s| (s - 0.25).abs() > 0.01),
        "active samples must carry noise_rms-level noise"
    );
    let quiet = render(vec![0.0_f32; 4_800]);
    assert!(
        quiet[0].iter().all(|s| s.abs() < 0.005),
        "silent samples must carry quiet-level noise only"
    );
}

#[test]
fn fake_gain_saturates_like_a_converter() {
    let wire = VirtualWire::new(FakeAudioConfig {
        gain: 4.0,
        tail_secs: 0.0,
        ..FakeAudioConfig::default()
    });
    let mut playback = wire.playback("fake");
    let mut recorder = wire.recorder("fake");

    let stimulus = vec![0.5_f32; 4_800];
    let temp_wav = super::write::write_temp_wav(&stimulus, SAMPLE_RATE, 1).unwrap();
    playback
        .start(temp_wav.path(), None, 0, SAMPLE_RATE)
        .unwrap();
    recorder.start(None, &[0], SAMPLE_RATE, 0.1).unwrap();
    let take = recorder.stop().unwrap();
    let peak = take.recorded_per_channel[0]
        .iter()
        .map(|s| s.abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(peak, 1.0);
}

#[test]
fn fake_capture_without_playback_is_silent() {
    let wire = VirtualWire::default();
    let mut recorder = wire.recorder("fake");
    recorder.start(None, &[0], SAMPLE_RATE, 0.1).unwrap();
    let take = recorder.stop().unwrap();
    assert!(take.recorded_per_channel[0].iter().all(|s| *s == 0.0));
}

#[test]
fn emulated_sweep_loopback_is_flat_and_trustworthy() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = super::write::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("recorded.wav");
    let csv = dir.path().join("analysis.csv");

    let wire = VirtualWire::default();
    let mut playback = wire.playback("record_and_analyze");
    let mut recorder = wire.recorder("record_and_analyze");
    let analysis = record_and_analyze_with_recorder(
        &mut playback,
        &mut recorder,
        temp_wav.path(),
        &recorded_wav,
        &sweep,
        SAMPLE_RATE,
        &csv,
        0,
        0,
        None,
        None,
        None,
        Some((100.0, 20_000.0)),
        1,
        None,
    )
    .unwrap();

    assert!(analysis.quality.trustworthy);
    assert_eq!(analysis.accepted_count, 1);
    assert_eq!(analysis.dropped_samples, 0);
    assert!(recorded_wav.exists());
    assert!(csv.exists());

    // Digital loopback: flat ~0 dB transfer, mirroring the e2e bounds.
    let band = band_spl(
        &analysis.result.frequencies,
        &analysis.result.spl_db,
        100.0,
        10_000.0,
    );
    assert!(!band.is_empty());
    let (mean, variation) = mean_and_variation(&band);
    assert!(
        (-3.0..=3.0).contains(&mean),
        "mean SPL {mean:.3} dB outside [-3, 3]"
    );
    assert!(variation < 1.0, "variation {variation:.3} dB too high");
    if let Some(drift) = &analysis.drift {
        assert!(
            drift.ppm.abs() <= super::quality::DRIFT_CORRECT_PPM,
            "no synthetic drift: expected |ppm| <= 20, got {:.2}",
            drift.ppm
        );
    }
}

#[test]
fn emulated_sweep_reports_no_drift_without_timing_chirps() {
    // Chirpless takes have no unbiased drift source: even with an injected
    // 150 ppm capture-clock stretch, the engine must report drift None
    // rather than attempt estimation (two-window drift never worked on
    // real takes — degenerate locks or IR bias; see
    // docs/bug-math-dsp-lag-f32-absorption.md). Plans with timing chirps
    // are the drift path.
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = super::write::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("recorded.wav");
    let csv = dir.path().join("analysis.csv");

    let wire = VirtualWire::new(FakeAudioConfig {
        latency_frames: 4_800,
        drift_ppm: 150.0,
        ..FakeAudioConfig::default()
    });
    let mut playback = wire.playback("record_and_analyze");
    let mut recorder = wire.recorder("record_and_analyze");
    let analysis = record_and_analyze_with_recorder(
        &mut playback,
        &mut recorder,
        temp_wav.path(),
        &recorded_wav,
        &sweep,
        SAMPLE_RATE,
        &csv,
        0,
        0,
        None,
        None,
        None,
        Some((100.0, 20_000.0)),
        1,
        None,
    )
    .unwrap();

    assert!(
        analysis.drift.is_none(),
        "chirpless take must skip drift estimation, got {:?}",
        analysis.drift
    );
    assert!(!analysis.drift_corrected);
}

#[test]
fn emulated_clipped_take_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = super::write::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("recorded.wav");
    let csv = dir.path().join("analysis.csv");

    let wire = VirtualWire::new(FakeAudioConfig {
        gain: 4.0,
        ..FakeAudioConfig::default()
    });
    let mut playback = wire.playback("record_and_analyze");
    let mut recorder = wire.recorder("record_and_analyze");
    let error = record_and_analyze_with_recorder(
        &mut playback,
        &mut recorder,
        temp_wav.path(),
        &recorded_wav,
        &sweep,
        SAMPLE_RATE,
        &csv,
        0,
        0,
        None,
        None,
        None,
        Some((100.0, 20_000.0)),
        1,
        None,
    )
    .unwrap_err();
    assert!(error.contains("clipped hard"), "{error}");
}

#[test]
fn emulated_silent_take_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = super::write::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("recorded.wav");
    let csv = dir.path().join("analysis.csv");

    let wire = VirtualWire::new(FakeAudioConfig {
        gain: 0.0,
        ..FakeAudioConfig::default()
    });
    let mut playback = wire.playback("record_and_analyze");
    let mut recorder = wire.recorder("record_and_analyze");
    let error = record_and_analyze_with_recorder(
        &mut playback,
        &mut recorder,
        temp_wav.path(),
        &recorded_wav,
        &sweep,
        SAMPLE_RATE,
        &csv,
        0,
        0,
        None,
        None,
        None,
        Some((100.0, 20_000.0)),
        1,
        None,
    )
    .unwrap_err();
    assert!(error.contains("appears silent"), "{error}");
}

#[test]
fn emulated_capture_honors_cancel_before_start() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = super::write::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("recorded.wav");
    let csv = dir.path().join("analysis.csv");

    let cancel = Arc::new(AtomicBool::new(true));
    let wire = VirtualWire::default();
    let mut playback = wire.playback("record_and_analyze");
    let mut recorder = wire.recorder("record_and_analyze");
    let error = record_and_analyze_with_recorder(
        &mut playback,
        &mut recorder,
        temp_wav.path(),
        &recorded_wav,
        &sweep,
        SAMPLE_RATE,
        &csv,
        0,
        0,
        None,
        None,
        None,
        Some((100.0, 20_000.0)),
        1,
        Some(cancel),
    )
    .unwrap_err();
    assert_eq!(error, "cancelled");
}

#[test]
fn emulated_multi_two_mics_agree() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = super::write::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let wav_paths = vec![dir.path().join("mic0.wav"), dir.path().join("mic1.wav")];
    let csv_paths = vec![dir.path().join("mic0.csv"), dir.path().join("mic1.csv")];

    let wire = VirtualWire::default();
    let mut playback = wire.playback("record_and_analyze_multi");
    let mut recorder = wire.recorder("record_and_analyze_multi");
    let analyses = record_and_analyze_with_multi_recorder(
        &mut playback,
        &mut recorder,
        temp_wav.path(),
        &wav_paths,
        &sweep,
        SAMPLE_RATE,
        &csv_paths,
        0,
        &[0, 1],
        None,
        None,
        &[None, None],
        Some((100.0, 20_000.0)),
        1,
        None,
    )
    .unwrap();

    assert_eq!(analyses.len(), 2);
    let mut means = Vec::new();
    for analysis in &analyses {
        assert!(analysis.quality.trustworthy);
        let band = band_spl(
            &analysis.result.frequencies,
            &analysis.result.spl_db,
            100.0,
            10_000.0,
        );
        let (mean, variation) = mean_and_variation(&band);
        assert!((-3.0..=3.0).contains(&mean), "mean {mean:.3}");
        assert!(variation < 1.0, "variation {variation:.3}");
        means.push(mean);
    }
    assert!(
        (means[0] - means[1]).abs() < 1.0,
        "cross-channel spread {:.3} dB",
        (means[0] - means[1]).abs()
    );
}

#[test]
fn emulated_overruns_thread_into_the_report() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = super::write::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("recorded.wav");
    let csv = dir.path().join("analysis.csv");

    let wire = VirtualWire::new(FakeAudioConfig {
        dropped_samples: 1_234,
        ..FakeAudioConfig::default()
    });
    let mut playback = wire.playback("record_and_analyze");
    let mut recorder = wire.recorder("record_and_analyze");
    let analysis = record_and_analyze_with_recorder(
        &mut playback,
        &mut recorder,
        temp_wav.path(),
        &recorded_wav,
        &sweep,
        SAMPLE_RATE,
        &csv,
        0,
        0,
        None,
        None,
        None,
        Some((100.0, 20_000.0)),
        1,
        None,
    )
    .unwrap();
    assert_eq!(analysis.dropped_samples, 1_234);
}

#[test]
fn emulated_rate_mismatch_resamples_the_reference() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = super::write::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("recorded.wav");
    let csv = dir.path().join("analysis.csv");

    let wire = VirtualWire::new(FakeAudioConfig {
        negotiated_input_rate: Some(44_100),
        ..FakeAudioConfig::default()
    });
    let mut playback = wire.playback("record_and_analyze");
    let mut recorder = wire.recorder("record_and_analyze");
    record_and_analyze_with_recorder(
        &mut playback,
        &mut recorder,
        temp_wav.path(),
        &recorded_wav,
        &sweep,
        SAMPLE_RATE,
        &csv,
        0,
        0,
        None,
        None,
        None,
        Some((100.0, 20_000.0)),
        1,
        None,
    )
    .unwrap();

    let reader = hound::WavReader::open(&recorded_wav).unwrap();
    assert_eq!(reader.spec().sample_rate, 44_100);
}

#[test]
fn wake_burst_is_deterministic_faded_and_bounded() {
    use super::wake::{WAKE_BURST_SECS, wake_burst_samples};

    let first = wake_burst_samples(SAMPLE_RATE);
    let second = wake_burst_samples(SAMPLE_RATE);
    assert_eq!(first, second);
    assert_eq!(first.len(), (WAKE_BURST_SECS * SAMPLE_RATE as f32) as usize);
    assert!(first.iter().all(|sample| sample.is_finite()));
    let peak = first.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    assert!((peak - 0.5).abs() < 1e-6, "{peak}");
    // Raised-cosine fades pin both ends near zero; the middle stays hot.
    assert!(first[0].abs() < 1e-4 && first[first.len() - 1].abs() < 1e-4);
    let mid_peak = first[first.len() / 4..3 * first.len() / 4]
        .iter()
        .map(|s| s.abs())
        .fold(0.0_f32, f32::max);
    assert!(mid_peak > 0.4, "{mid_peak}");
}

#[test]
fn wake_up_rides_through_sleeping_first_starts() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = super::write::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("recorded.wav");
    let csv = dir.path().join("analysis.csv");

    // The first two input starts render silence (USB asleep); the wake-up
    // burns them and the real take records clean.
    let config = FakeAudioConfig {
        dead_starts: 2,
        ..FakeAudioConfig::default()
    };
    let wire = VirtualWire::new(config);
    let mut playback = wire.playback("record_and_analyze");
    let mut recorder = wire.recorder("record_and_analyze");
    let analysis = record_and_analyze_with_recorder(
        &mut playback,
        &mut recorder,
        temp_wav.path(),
        &recorded_wav,
        &sweep,
        SAMPLE_RATE,
        &csv,
        0,
        0,
        None,
        None,
        None,
        Some((100.0, 20_000.0)),
        1,
        None,
    )
    .expect("wake-up must absorb the sleeping starts");
    assert!(analysis.quality.trustworthy);
    assert_eq!(analysis.accepted_count, 1);
}

#[test]
fn wake_up_budget_exhaustion_proceeds_without_error() {
    use super::wake::wake_audio_chain;
    use std::time::{Duration, Instant};

    // A chain that never answers: the budget bounds the wait and the call
    // still returns Ok — the take gates own the refusal.
    let config = FakeAudioConfig {
        dead_starts: usize::MAX,
        ..FakeAudioConfig::default()
    };
    let wire = VirtualWire::new(config);
    let mut playback = wire.playback("wake");
    let mut recorder = wire.recorder("wake");
    let started = Instant::now();
    wake_audio_chain(
        &mut playback,
        &mut recorder,
        SAMPLE_RATE,
        0,
        0,
        None,
        None,
        None,
        Duration::from_millis(300),
        "wake",
    )
    .expect("exhaustion must proceed, not fail");
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "budget must bound the wait"
    );
}

#[test]
fn wake_up_honors_cancel_immediately() {
    use super::wake::wake_audio_chain;
    use std::time::Duration;

    let wire = VirtualWire::default();
    let mut playback = wire.playback("wake");
    let mut recorder = wire.recorder("wake");
    let cancel = Arc::new(AtomicBool::new(true));
    let error = wake_audio_chain(
        &mut playback,
        &mut recorder,
        SAMPLE_RATE,
        0,
        0,
        None,
        None,
        Some(&cancel),
        Duration::from_secs(5),
        "wake",
    )
    .unwrap_err();
    assert_eq!(error, "cancelled");
}
