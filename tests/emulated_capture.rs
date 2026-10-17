//! Emulated end-to-end capture through the public API only.
//!
//! Mirrors the hardware `AEQ_E2E` loopback bounds (flat ~0 dB transfer)
//! with no audio device, proving downstream crates can emulate capture
//! via `record_and_analyze_with_recorder` plus the `test-util` fakes.

// Rust guideline compliant 2026-02-21

#![cfg(feature = "test-util")]

use sotf_capture::signal_recorder::{
    FakeAudioConfig, FakeAudioProfile, VirtualWire, record_and_analyze_with_multi_recorder,
    record_and_analyze_with_recorder,
};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

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

#[expect(
    clippy::cast_precision_loss,
    reason = "Fixture band lengths normalize floating-point statistics over small test buffers."
)]
fn band_mean_variation(frequencies: &[f32], spl_db: &[f32]) -> (f32, f32) {
    let band: Vec<f32> = frequencies
        .iter()
        .zip(spl_db.iter())
        .filter(|(freq, _)| (100.0..=10_000.0).contains(*freq))
        .map(|(_, spl)| *spl)
        .collect();
    assert_ne!(band, [] as [f32; 0]);
    let mean = band.iter().sum::<f32>() / band.len() as f32;
    let min = band.iter().copied().fold(f32::INFINITY, f32::min);
    let max = band.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (mean, max - min)
}

fn analyze_profile(
    profile: FakeAudioProfile,
) -> Result<sotf_capture::signal_recorder::CaptureAnalysis, String> {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = sotf_capture::signal_recorder::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("recorded.wav");
    let csv = dir.path().join("analysis.csv");
    let wire = VirtualWire::for_profile(profile);
    let mut playback = wire.playback("profile_capture");
    let mut recorder = wire.recorder("profile_capture");
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

#[test]
fn public_surface_multi_take_capture_reports_coherent_quality() {
    let dir = tempfile::tempdir().unwrap();
    let mut sweep = vec![0.0; SAMPLE_RATE as usize / 10];
    sweep.extend(sweep_2s());
    let temp_wav = sotf_capture::signal_recorder::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let wav_paths = vec![dir.path().join("mic0.wav"), dir.path().join("mic1.wav")];
    let csv_paths = vec![dir.path().join("mic0.csv"), dir.path().join("mic1.csv")];

    let wire = VirtualWire::for_profile(FakeAudioProfile::Clean);
    let mut playback = wire.playback("record_and_analyze_multi_repeat");
    let mut recorder = wire.recorder("record_and_analyze_multi_repeat");
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
        4,
        None,
    )
    .unwrap();

    assert_eq!(analyses.len(), 2);
    for (analysis, csv_path) in analyses.iter().zip(&csv_paths) {
        assert_eq!(analysis.accepted_count, 4);
        assert!(
            analysis.quality.trustworthy,
            "{:?}",
            analysis.quality.issues
        );
        assert!(analysis.quality.mean_coherence.is_some_and(|coherence| {
            coherence.is_finite() && (0.0..=1.0).contains(&coherence)
        }));
        assert!(analysis.quality.median_snr_db.is_some_and(f32::is_finite));
        // Coherent-repeat metrics live in the full quality summary. The
        // THD advisory remains separately represented as a typed finding.
        assert!(analysis.quality_findings.iter().any(|finding| {
            finding.metric == sotf_capture::quality::QualityMetric::TotalHarmonicDistortion
        }));
        let csv = std::fs::read_to_string(csv_path).unwrap();
        let header = csv.lines().next().unwrap();
        assert!(header.contains("coherence"));
        assert!(header.contains("noise_floor_db"));
    }
}

#[test]
fn public_repeat_capture_keeps_digital_silence_snr_unknown() {
    use sotf_capture::quality::{QualityMetric, QualityPolicy, QualityStatus};

    let dir = tempfile::tempdir().unwrap();
    let mut sweep = vec![0.0; SAMPLE_RATE as usize / 10];
    sweep.extend(sweep_2s());
    let temp_wav = sotf_capture::signal_recorder::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("recorded.wav");
    let csv = dir.path().join("analysis.csv");
    let wire = VirtualWire::for_profile(FakeAudioProfile::UnknownEvidence);
    let mut playback = wire.playback("repeat_unknown_snr");
    let mut recorder = wire.recorder("repeat_unknown_snr");

    let capture = record_and_analyze_with_recorder(
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
        3,
        None,
    )
    .expect("three exact-silence takes still produce a response analysis");

    let snr = capture
        .quality_findings
        .iter()
        .find(|finding| finding.metric == QualityMetric::BroadbandSnr)
        .expect("repeat-capture SNR finding");
    assert_eq!(snr.policy, QualityPolicy::SweepAnalysis);
    assert_eq!(snr.status, QualityStatus::Unknown);
    assert_eq!(
        snr.measured, None,
        "unknown evidence is never encoded as 0 dB"
    );
    assert_eq!(snr.threshold.unwrap().value, 10.0);
    assert!(
        snr.unknown_reason
            .as_deref()
            .unwrap()
            .contains("digital zeros")
    );
    assert!(!capture.quality.trustworthy);

    let csv_header = std::fs::read_to_string(&csv).unwrap();
    let header = csv_header.lines().next().unwrap();
    assert!(
        !header.split(',').any(|column| column == "noise_floor_db"),
        "exact digital silence is not serialized as a measured noise floor: {header}"
    );
}

#[test]
fn public_surface_cancellation_and_device_start_errors_reach_callers() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = sotf_capture::signal_recorder::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("recorded.wav");
    let csv = dir.path().join("analysis.csv");

    let wire = VirtualWire::default();
    let mut playback = wire.playback("cancelled_capture");
    let mut recorder = wire.recorder("cancelled_capture");
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
        Some(Arc::new(AtomicBool::new(true))),
    )
    .unwrap_err();
    assert_eq!(error, "cancelled");

    let wire = VirtualWire::new(FakeAudioConfig {
        fail_input_start: Some("injected input start failure".to_string()),
        ..FakeAudioConfig::default()
    });
    let mut playback = wire.playback("device_error_capture");
    let mut recorder = wire.recorder("device_error_capture");
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
    assert!(error.contains("injected input start failure"), "{error}");
}

#[test]
fn public_profile_emulates_distortion_noise_and_unknown_evidence() {
    let distorted = analyze_profile(FakeAudioProfile::HighThd).unwrap();
    let thd = distorted
        .quality_findings
        .iter()
        .find(|finding| {
            finding.metric == sotf_capture::quality::QualityMetric::TotalHarmonicDistortion
        })
        .expect("THD finding");
    assert_eq!(thd.status, sotf_capture::quality::QualityStatus::Warning);
    assert_eq!(
        thd.method,
        sotf_capture::quality::QualityMethod::FarinaHarmonicImpulseResponse
    );
    assert!(thd.measured.unwrap() > 10.0);
    assert!(thd.band.is_some());
    assert_eq!(thd.unit, sotf_capture::quality::QualityUnit::Percent);
    assert_eq!(
        thd.policy,
        sotf_capture::quality::QualityPolicy::SweepAnalysis
    );
    assert!(!distorted.quality.trustworthy);

    let noisy = analyze_profile(FakeAudioProfile::LowSnr).unwrap();
    let snr = noisy
        .quality_findings
        .iter()
        .find(|finding| finding.metric == sotf_capture::quality::QualityMetric::BroadbandSnr)
        .expect("single-take SNR finding");
    assert_eq!(
        snr.status,
        sotf_capture::quality::QualityStatus::Warning,
        "measured {:?}",
        snr.measured
    );
    assert!(
        snr.measured.unwrap() < 10.0,
        "measured {} dB",
        snr.measured.unwrap()
    );
    assert_eq!(snr.threshold.unwrap().value, 10.0);
    assert!(!noisy.quality.trustworthy);

    let unknown = analyze_profile(FakeAudioProfile::UnknownEvidence).unwrap();
    let snr = unknown
        .quality_findings
        .iter()
        .find(|finding| finding.metric == sotf_capture::quality::QualityMetric::BroadbandSnr)
        .expect("unknown SNR finding");
    assert_eq!(snr.status, sotf_capture::quality::QualityStatus::Unknown);
    assert_eq!(snr.measured, None);
    assert_eq!(snr.threshold.unwrap().value, 10.0);
    assert!(snr.unknown_reason.as_deref().unwrap().contains("silent"));
}

#[test]
fn clean_profile_measures_snr_for_short_narrow_octave_sweep() {
    use sotf_capture::quality::{QualityMetric, QualityPolicy, QualityStatus, QualityUnit};
    use sotf_capture::signal_recorder::{
        SignalType, prepare_measurement_signal, sweep_params_from_config,
    };

    let dir = tempfile::tempdir().unwrap();
    let params = sweep_params_from_config(1_000.0, 2_000.0, 0.5, Some(1.0), None, None);
    let sweep = prepare_measurement_signal(SignalType::Sweep, &params, 1.0, SAMPLE_RATE).unwrap();
    let temp_wav = sotf_capture::signal_recorder::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let recorded_wav = dir.path().join("clean-narrow.wav");
    let csv = dir.path().join("clean-narrow.csv");
    let wire = VirtualWire::for_profile(FakeAudioProfile::Clean);
    let mut playback = wire.playback("clean_narrow_capture");
    let mut recorder = wire.recorder("clean_narrow_capture");
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
        Some((1_000.0, 2_000.0)),
        1,
        None,
    )
    .expect("single narrow octave capture should analyze cleanly");

    let snr = analysis
        .quality_findings
        .iter()
        .find(|finding| {
            finding.metric == QualityMetric::BroadbandSnr
                && finding.policy == QualityPolicy::SingleTake
        })
        .expect("single-take SNR finding");
    assert_eq!(snr.status, QualityStatus::Pass, "{snr:?}");
    assert_eq!(snr.unit, QualityUnit::Decibels);
    assert_eq!(snr.threshold.unwrap().value, 10.0);
    assert!(snr.measured.unwrap() > 30.0, "{snr:?}");
    assert_eq!(snr.unknown_reason, None);
    let thd = analysis
        .quality_findings
        .iter()
        .find(|finding| finding.metric == QualityMetric::TotalHarmonicDistortion)
        .expect("sweep THD finding");
    assert_eq!(thd.status, QualityStatus::Pass, "{thd:?}");
    let band = thd.band.unwrap();
    assert_eq!(band.low_hz, 1_000.0);
    assert_eq!(band.high_hz, 2_000.0);
    assert!(
        thd.measured.unwrap() <= thd.threshold.unwrap().value,
        "{thd:?}"
    );
    assert!(
        analysis.quality.trustworthy,
        "{:?}",
        analysis.quality_findings
    );
}

#[test]
fn public_profile_applies_distinct_room_impulse_response_per_input_channel() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = sweep_2s();
    let temp_wav = sotf_capture::signal_recorder::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
    let wav_paths = vec![dir.path().join("quiet.wav"), dir.path().join("loud.wav")];
    let csv_paths = vec![dir.path().join("quiet.csv"), dir.path().join("loud.csv")];
    let mut config = FakeAudioConfig::for_profile(FakeAudioProfile::Clean);
    config.channel_impulse_responses = vec![vec![0.5], vec![1.0]];
    let wire = VirtualWire::new(config);
    let mut playback = wire.playback("room_response");
    let mut recorder = wire.recorder("room_response");
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
    let first = band_mean_variation(&analyses[0].result.frequencies, &analyses[0].result.spl_db).0;
    let second = band_mean_variation(&analyses[1].result.frequencies, &analyses[1].result.spl_db).0;
    assert!(
        (second - first - 6.02).abs() < 0.2,
        "channel gain delta was {:.2} dB",
        second - first
    );
}

fn spl_at_nearest(frequencies: &[f32], spl_db: &[f32], target_hz: f32) -> f32 {
    frequencies
        .iter()
        .zip(spl_db.iter())
        .min_by(|(a, _), (b, _)| (*a - target_hz).abs().total_cmp(&(*b - target_hz).abs()))
        .map(|(_, spl)| *spl)
        .expect("analysis grid must not be empty")
}

#[test]
fn synthetic_room_comb_and_mode_match_ground_truth() {
    // Ground truth is defined analytically, independently of the analysis
    // under test: a reflection 5 ms after the direct path produces a comb
    // with peaks at multiples of 200 Hz and notches halfway between them;
    // a decaying 120 Hz resonance produces a narrow local peak. The
    // deconvolved magnitude must reproduce both features.
    fn analyze_room(
        impulse_response: Vec<f32>,
        latency_frames: usize,
    ) -> sotf_capture::signal_recorder::CaptureAnalysis {
        let dir = tempfile::tempdir().unwrap();
        let sweep = sotf_capture::signal_recorder::generate_signal(
            sotf_capture::signal_recorder::SignalType::Sweep,
            &sotf_capture::signal_recorder::SignalParams::Sweep {
                start_freq: 50.0,
                end_freq: 20_000.0,
                amp: 0.5,
            },
            2.0,
            SAMPLE_RATE,
        )
        .unwrap();
        let temp_wav =
            sotf_capture::signal_recorder::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
        let recorded_wav = dir.path().join("room.wav");
        let csv = dir.path().join("room.csv");
        let wire = VirtualWire::new(FakeAudioConfig {
            latency_frames,
            channel_impulse_responses: vec![impulse_response],
            ..FakeAudioConfig::for_profile(FakeAudioProfile::Clean)
        });
        let mut playback = wire.playback("synthetic_room");
        let mut recorder = wire.recorder("synthetic_room");
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
            Some((50.0, 20_000.0)),
            1,
            None,
        )
        .expect("synthetic room capture must analyze")
    }

    // Room A: direct path plus one 0.7 reflection at 5 ms.
    // Theory: peak |1 + 0.7| = +4.6 dB, notch |1 - 0.7| = -10.5 dB.
    let mut comb_ir = vec![0.0f32; 1_024];
    comb_ir[0] = 1.0;
    comb_ir[240] = 0.7; // 240 samples = 5 ms at 48 kHz
    let comb = analyze_room(comb_ir, 480);
    assert!(comb.quality.trustworthy, "{:?}", comb.quality.issues);
    assert!(
        (comb.result.estimated_lag_samples - 480).abs() <= 2,
        "direct-path lag {} must ignore the later reflection",
        comb.result.estimated_lag_samples
    );
    let notch_db = spl_at_nearest(&comb.result.frequencies, &comb.result.spl_db, 300.0);
    let peak_db = spl_at_nearest(&comb.result.frequencies, &comb.result.spl_db, 400.0);
    assert!(
        peak_db - notch_db > 8.0,
        "comb contrast {:.1} dB (peak {peak_db:.1}, notch {notch_db:.1}); theory predicts ~15 dB",
        peak_db - notch_db
    );

    // Room B: flat direct path plus a decaying 120 Hz mode scaled for a
    // ~+6 dB resonant contribution (peak of A * tau * fs / 2 ~ 1.0). The
    // cosine starts in phase with the direct path so the contributions add
    // constructively at resonance, and tau keeps the resonance wider than
    // the analysis smoothing so the peak survives fractional-octave
    // averaging (bandwidth ~1/(pi * tau) ~ 6 Hz).
    let fs = SAMPLE_RATE as f32;
    let tau_s = 0.05f32;
    let amplitude = 2.0 / (tau_s * fs);
    let mut modal_ir = vec![0.0f32; SAMPLE_RATE as usize / 2];
    modal_ir[0] = 1.0;
    for (n, tap) in modal_ir.iter_mut().enumerate() {
        let t = n as f32 / fs;
        *tap += amplitude * (-t / tau_s).exp() * (2.0 * std::f32::consts::PI * 120.0 * t).cos();
    }
    let modal = analyze_room(modal_ir, 0);
    assert!(modal.quality.trustworthy, "{:?}", modal.quality.issues);
    let mode_db = spl_at_nearest(&modal.result.frequencies, &modal.result.spl_db, 120.0);
    let flank_db = 0.5
        * (spl_at_nearest(&modal.result.frequencies, &modal.result.spl_db, 80.0)
            + spl_at_nearest(&modal.result.frequencies, &modal.result.spl_db, 180.0));
    assert!(
        mode_db - flank_db > 3.0,
        "modal peak {mode_db:.1} dB vs flanks {flank_db:.1} dB; theory predicts a clear local peak"
    );
}

#[test]
fn microphone_compensation_subtracts_the_calibration_deviation() {
    // Ground truth: the calibration file declares the microphone reads
    // +6 dB hot between 1 kHz and 4 kHz. The persisted CSV is the
    // compensated artifact (the in-memory result stays raw by design), so
    // on a flat wire the CSV magnitude in the middle of that plateau must
    // sit 6 dB below the uncompensated CSV, while a band outside the
    // plateau is unchanged. The expectation comes from the file contents,
    // not from the analysis under test.
    fn analyze_with_calibration(calibration_path: Option<&str>) -> Vec<(f32, f32)> {
        let dir = tempfile::tempdir().unwrap();
        let sweep = sweep_2s();
        let temp_wav =
            sotf_capture::signal_recorder::write_temp_wav(&sweep, SAMPLE_RATE, 1).unwrap();
        let recorded_wav = dir.path().join("recorded.wav");
        let csv = dir.path().join("analysis.csv");
        let wire = VirtualWire::for_profile(FakeAudioProfile::Clean);
        let mut playback = wire.playback("calibration_golden");
        let mut recorder = wire.recorder("calibration_golden");
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
            calibration_path,
            Some((100.0, 20_000.0)),
            1,
            None,
        )
        .expect("flat-wire capture must analyze");
        let text = std::fs::read_to_string(&csv).unwrap();
        text.lines()
            .skip(1)
            .filter_map(|line| {
                let mut columns = line.split(',');
                let freq: f32 = columns.next()?.parse().ok()?;
                let spl: f32 = columns.next()?.parse().ok()?;
                Some((freq, spl))
            })
            .collect()
    }

    let dir = tempfile::tempdir().unwrap();
    let calibration = dir.path().join("mic-cal.txt");
    std::fs::write(
        &calibration,
        "# synthetic microphone deviation: +6 dB plateau 1-4 kHz, flat elsewhere\n\
         100 0.0\n900 0.0\n1000 6.0\n4000 6.0\n4500 0.0\n20000 0.0\n",
    )
    .unwrap();

    let plain = analyze_with_calibration(None);
    let compensated = analyze_with_calibration(Some(calibration.to_str().unwrap()));

    let band_mean = |rows: &[(f32, f32)], low: f32, high: f32| {
        let band: Vec<f32> = rows
            .iter()
            .filter(|(freq, _)| (low..=high).contains(freq))
            .map(|(_, spl)| *spl)
            .collect();
        assert!(!band.is_empty(), "empty band {low}-{high} Hz");
        #[expect(
            clippy::cast_precision_loss,
            reason = "Band lengths normalize floating-point statistics over small test buffers."
        )]
        let mean = band.iter().sum::<f32>() / band.len() as f32;
        mean
    };

    let plateau_delta =
        band_mean(&compensated, 1_500.0, 3_000.0) - band_mean(&plain, 1_500.0, 3_000.0);
    assert!(
        (plateau_delta + 6.0).abs() < 1.0,
        "compensated plateau delta {plateau_delta:.2} dB, expected -6 +/- 1 dB"
    );
    let outside_delta = band_mean(&compensated, 150.0, 400.0) - band_mean(&plain, 150.0, 400.0);
    assert!(
        outside_delta.abs() < 1.0,
        "outside-plateau delta {outside_delta:.2} dB, expected ~0 dB"
    );
}

#[test]
fn public_profile_clipping_is_refused_by_the_capture_gate() {
    let clipped = analyze_profile(FakeAudioProfile::Clipping).unwrap();
    assert!(clipped.quality.clipping.clipped_samples > 0);
    assert!(clipped.quality_findings.iter().any(|finding| {
        finding.metric == sotf_capture::quality::QualityMetric::Clipping
            && finding.status == sotf_capture::quality::QualityStatus::Warning
            && finding.measured.unwrap() > 0.0
    }));
}
