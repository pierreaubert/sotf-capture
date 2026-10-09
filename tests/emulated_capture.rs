//! Emulated end-to-end capture through the public API only.
//!
//! Mirrors the hardware `AEQ_E2E` loopback bounds (flat ~0 dB transfer)
//! with no audio device, proving downstream crates can emulate capture
//! via `record_and_analyze_with_recorder` plus the `test-util` fakes.

// Rust guideline compliant 2026-02-21

#![cfg(feature = "test-util")]

use sotf_capture::signal_recorder::{
    FakeAudioConfig, VirtualWire, record_and_analyze_with_multi_recorder,
    record_and_analyze_with_recorder,
};

const SAMPLE_RATE: u32 = 48_000;

fn sweep_2s() -> Vec<f32> {
    sotf_capture::signal_recorder::generate_signal(
        sotf_capture::signal_recorder::SignalType::Sweep,
        &sotf_capture::signal_recorder::SignalParams::Sweep {
            start_freq: 100.0,
            end_freq: 20_000.0,
            amp: 0.5,
        },
        2.0,
        SAMPLE_RATE,
    )
    .unwrap()
}

fn band_mean_variation(frequencies: &[f32], spl_db: &[f32]) -> (f32, f32) {
    let band: Vec<f32> = frequencies
        .iter()
        .zip(spl_db.iter())
        .filter(|(freq, _)| (100.0..=10_000.0).contains(*freq))
        .map(|(_, spl)| *spl)
        .collect();
    assert!(!band.is_empty());
    let mean = band.iter().sum::<f32>() / band.len() as f32;
    let min = band.iter().copied().fold(f32::INFINITY, f32::min);
    let max = band.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (mean, max - min)
}

#[test]
fn public_surface_emulated_sweep_loopback() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = sotf_capture::signal_recorder::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
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
    let (mean, variation) =
        band_mean_variation(&analysis.result.frequencies, &analysis.result.spl_db);
    assert!(
        (-3.0..=3.0).contains(&mean),
        "mean SPL {mean:.3} dB outside [-3, 3]"
    );
    assert!(variation < 1.0, "variation {variation:.3} dB too high");
}

#[test]
fn public_surface_emulated_two_mic_capture() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = sotf_capture::signal_recorder::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let wav_paths = vec![dir.path().join("mic0.wav"), dir.path().join("mic1.wav")];
    let csv_paths = vec![dir.path().join("mic0.csv"), dir.path().join("mic1.csv")];

    let wire = VirtualWire::new(FakeAudioConfig {
        latency_frames: 480,
        ..FakeAudioConfig::default()
    });
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
    for analysis in &analyses {
        assert!(analysis.quality.trustworthy);
        assert!(
            (analysis.result.estimated_lag_samples - 480).abs() <= 2,
            "lag {}, expected ~480",
            analysis.result.estimated_lag_samples
        );
    }
}
