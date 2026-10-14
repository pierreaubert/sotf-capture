//! Calibrated shared-origin artifacts, without arrival alignment.

// Rust guideline compliant 2026-02-21
use super::{CaptureSessionPlan, protocol::CaptureStimulus};
use math_audio_dsp::analysis::{MicrophoneCompensation, deconvolve_common_clock};
use serde::{Deserialize, Serialize};
use std::{fmt::Write, path::Path};

/// IR-only speech-intelligibility QA for one take.
///
/// Indirect STI from the deconvolved IR: no ambient-noise, hearing-threshold,
/// or masking corrections. Advisory only, never IEC instrument certification.
///
/// The companion direct score comes from recording the `stipa` stimulus and
/// scoring it with `math_rir::sti_direct::analyze_stipa_direct` (10 s minimum
/// after edge trim, loopback reference preferred; without one the received
/// depths are compared against the generator's 0.55 sent depth). The
/// generate→analyze round-trip is pinned by
/// `signal_recorder::tests::stipa_generate_analyze_*`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureStiReport {
    /// Speech Transmission Index in [0, 1].
    pub sti: f64,
    /// Per-octave modulation transfer indices, 125 Hz to 8 kHz.
    pub mti: [f64; 7],
    /// IR span analysed, in seconds.
    pub duration_s: f64,
}

/// Calibrated phase and IR artifacts with an explicit timing bandwidth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturePhaseArtifacts {
    /// Relative CSV path containing frequency, magnitude and measured phase.
    pub response_file: String,
    /// Relative JSON path containing the band-limited, periodic real IR.
    pub impulse_file: String,
    /// Relative `time_ms,amplitude` CSV of the same IR for `RoomEQ` measured-IR
    /// declarations. Absent on artifacts written before CSV export existed.
    #[serde(default)]
    pub impulse_csv_file: Option<String>,
    /// IR-only STI QA; absent when the IR cannot support the analysis
    /// (short span, missing band energy, or unsuitable rate).
    #[serde(default)]
    pub sti: Option<CaptureStiReport>,
    /// Relative `frequency_hz,coherence` CSV of Welch magnitude-squared
    /// coherence between the take and its reference; absent when the pair
    /// is too short or when written before coherence export existed.
    #[serde(default)]
    pub coherence_file: Option<String>,
    /// Upper frequency for at most nine degrees of conditional clock error.
    pub timing_limit_hz: f64,
    /// Lower calibrated sweep frequency.
    pub low_hz: f64,
    /// Upper calibrated sweep frequency; not itself a coherent-use guarantee.
    pub high_hz: f64,
}

/// Indirect STI QA for one deconvolved IR, or `None` when unsuitable.
///
/// Suitability is decided by the analyzer itself (span, band energy, rate);
/// an unsuitable IR keeps its phase artifacts and only loses the advisory
/// score, with the reason logged for the operator.
#[expect(
    clippy::cast_possible_truncation,
    reason = "The STI estimator accepts f32 audio, while the saved impulse response uses f64."
)]
fn take_sti(impulse_response: &[f64], sample_rate_hz: u32) -> Option<CaptureStiReport> {
    let linear: Vec<f32> = impulse_response
        .iter()
        .map(|sample| *sample as f32)
        .collect();
    match math_rir::sti::analyze_sti(&linear, f64::from(sample_rate_hz)) {
        Ok(result) => Some(CaptureStiReport {
            sti: result.sti,
            mti: result.mti,
            duration_s: result.duration_s,
        }),
        Err(error) => {
            log::warn!("take STI unavailable: {error:?}");
            None
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep the ordered acquisition and artifact-processing phases together so cleanup and provenance remain explicit."
)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "FFT indices use f64 time and frequency coordinates; calibration interpolation accepts f32 frequencies."
)]
pub(super) fn write_phase(
    samples: &[f32],
    stimulus: &CaptureStimulus,
    plan: &CaptureSessionPlan,
    calibration: &Path,
    output: &Path,
    index: usize,
    residual_us: f64,
) -> Result<CapturePhaseArtifacts, String> {
    if !residual_us.is_finite() || residual_us <= 0.0 {
        return Err("phase requires a finite positive timing bound".into());
    }
    let layout = &stimulus.layout;
    let end = layout.end_chirp_offset;
    // Identical origins retain acoustic travel time. Only the sweep and its
    // decay enter the FFT; neither timing chirp belongs to this source response.
    let capture = samples
        .get(layout.sweep_offset..end)
        .ok_or("missing sweep decay")?;
    let reference = stimulus
        .samples
        .get(layout.sweep_offset..end)
        .ok_or("invalid stimulus window")?;
    let coherence_file =
        match super::coherence::spectral_coherence(capture, reference, plan.sample_rate_hz) {
            Ok(grid) => {
                let name = format!("take-{index:03}-coherence.csv");
                let mut text = String::from("frequency_hz,coherence\n");
                for (frequency, gamma) in &grid {
                    writeln!(text, "{frequency},{gamma}").map_err(|e| e.to_string())?;
                }
                std::fs::write(output.join(&name), text).map_err(|e| e.to_string())?;
                Some(name)
            }
            Err(error) => {
                log::warn!("take coherence unavailable: {error}");
                None
            }
        };
    let capture: Vec<_> = capture.iter().map(|&x| f64::from(x)).collect();
    let reference: Vec<_> = reference.iter().map(|&x| f64::from(x)).collect();
    let response = deconvolve_common_clock(&capture, &reference, plan.sample_rate_hz)?;
    let compensation = MicrophoneCompensation::from_file(calibration)?;
    let low_hz = plan.sweep.start_hz.max(f64::from(
        *compensation
            .frequencies
            .first()
            .ok_or("empty calibration")?,
    ));
    let high_hz = plan.sweep.end_hz.min(f64::from(
        *compensation.frequencies.last().ok_or("empty calibration")?,
    ));
    if low_hz >= high_hz {
        return Err("calibration does not cover the sweep".into());
    }
    let bin_hz = f64::from(plan.sample_rate_hz) / response.impulse_response.len() as f64;
    let gains: Vec<_> = (0..response.spectrum.len())
        .map(|i| {
            let frequency = i as f64 * bin_hz;
            if frequency < low_hz || frequency > high_hz {
                0.0
            } else {
                10.0_f64.powf(-f64::from(compensation.interpolate_at(frequency as f32)) / 20.0)
            }
        })
        .collect();
    let response = response.with_magnitude_gains(&gains)?;
    let mut csv = String::from("frequency_hz,spl_db,phase_deg\n");
    let mut rows = 0;
    for (i, bin) in response.spectrum.iter().enumerate() {
        let frequency = i as f64 * bin_hz;
        if frequency < low_hz || frequency > high_hz {
            continue;
        }
        let magnitude = 20.0 * bin.norm().log10();
        if !magnitude.is_finite() {
            return Err("phase response contains an unexcited bin".into());
        }
        writeln!(csv, "{frequency},{magnitude},{}", bin.arg().to_degrees())
            .map_err(|e| e.to_string())?;
        rows += 1;
    }
    if rows < 2 {
        return Err("phase response has too few calibrated bins".into());
    }
    let impulse_csv_file = format!("take-{index:03}-ir.csv");
    let artifacts = CapturePhaseArtifacts {
        response_file: format!("take-{index:03}-phase.csv"),
        impulse_file: format!("take-{index:03}-ir.json"),
        impulse_csv_file: Some(impulse_csv_file.clone()),
        sti: take_sti(&response.impulse_response, plan.sample_rate_hz),
        coherence_file,
        timing_limit_hz: (25_000.0 / residual_us).min(high_hz),
        low_hz,
        high_hz,
    };
    let mut ir_csv = String::from("time_ms,amplitude\n");
    for (n, sample) in response.impulse_response.iter().enumerate() {
        writeln!(
            ir_csv,
            "{},{}",
            n as f64 * 1000.0 / f64::from(plan.sample_rate_hz),
            sample
        )
        .map_err(|e| e.to_string())?;
    }
    let impulse = serde_json::json!({
        "sample_rate_hz": plan.sample_rate_hz,
        "time_origin": "shared_stimulus_sweep_start",
        "low_hz": low_hz,
        "high_hz": high_hz,
        "timing_limit_hz": artifacts.timing_limit_hz,
        "microphone_phase_calibrated": false,
        "periodic_band_limited": true,
        "samples": response.impulse_response,
    });
    std::fs::write(output.join(&artifacts.response_file), csv).map_err(|e| e.to_string())?;
    std::fs::write(
        output.join(&artifacts.impulse_file),
        serde_json::to_vec(&impulse).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(output.join(&impulse_csv_file), ir_csv).map_err(|e| e.to_string())?;
    Ok(artifacts)
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Synthetic fixtures use bounded sample counts and intentionally quantized audio; exact comparisons verify unchanged settings and samples."
)]
mod tests {
    use super::*;

    #[test]
    fn impulse_csv_matches_json_samples_on_exact_time_grid() {
        let mut plan: CaptureSessionPlan =
            serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json"))
                .unwrap();
        plan.sweep.duration_secs = 0.25;
        let stimulus =
            super::super::protocol::prepare_capture_stimulus(&plan.clone().validate().unwrap())
                .unwrap();
        let output = tempfile::tempdir().unwrap();
        let flat = output.path().join("flat.txt");
        std::fs::write(&flat, "10 0\n24000 0\n").unwrap();
        let mut recorded = vec![0.0; stimulus.samples.len()];
        recorded[37..].copy_from_slice(&stimulus.samples[..stimulus.samples.len() - 37]);
        let artifacts =
            write_phase(&recorded, &stimulus, &plan, &flat, output.path(), 4, 20.0).unwrap();
        let csv_file = artifacts.impulse_csv_file.unwrap();
        assert_eq!(csv_file, "take-004-ir.csv");
        let sti = artifacts.sti.as_ref().unwrap();
        assert!((0.0..=1.0).contains(&sti.sti), "{}", sti.sti);
        assert!(sti.sti > 0.9, "{}", sti.sti);
        let coherence_file = artifacts.coherence_file.as_ref().unwrap();
        assert_eq!(coherence_file, "take-004-coherence.csv");
        let coherence = std::fs::read_to_string(output.path().join(coherence_file)).unwrap();
        let mut lines = coherence.lines();
        assert_eq!(lines.next(), Some("frequency_hz,coherence"));
        let rows: Vec<&str> = lines.collect();
        assert_ne!(rows, [] as [&str; 0]);
        let mut previous = -1.0;
        let mut sum = 0.0;
        for row in &rows {
            let (frequency, gamma) = row.split_once(',').unwrap();
            let frequency: f64 = frequency.parse().unwrap();
            let gamma: f64 = gamma.parse().unwrap();
            assert!(frequency > previous);
            assert!((0.0..=1.0).contains(&gamma), "{row}");
            previous = frequency;
            sum += gamma;
        }
        // A clean delayed copy of the reference coheres almost perfectly.
        assert!(sum / rows.len() as f64 > 0.99);
        let csv = std::fs::read_to_string(output.path().join(&csv_file)).unwrap();
        let ir: serde_json::Value = serde_json::from_slice(
            &std::fs::read(output.path().join(&artifacts.impulse_file)).unwrap(),
        )
        .unwrap();
        let samples = ir["samples"].as_array().unwrap();
        let mut lines = csv.lines();
        assert_eq!(lines.next(), Some("time_ms,amplitude"));
        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len(), samples.len());
        let step_ms = 1000.0 / f64::from(plan.sample_rate_hz);
        for (n, (row, expected)) in rows.iter().zip(samples).enumerate() {
            let (time, amplitude) = row.split_once(',').unwrap();
            let time: f64 = time.parse().unwrap();
            let amplitude: f64 = amplitude.parse().unwrap();
            assert!(
                (time - n as f64 * step_ms).abs() < 1e-9,
                "row {n} time {time} is off the declared sample grid"
            );
            assert_eq!(amplitude, expected.as_f64().unwrap());
        }
        // Legacy artifacts without the CSV field still parse.
        let legacy: CapturePhaseArtifacts = serde_json::from_str(
            r#"{"response_file":"take-000-phase.csv","impulse_file":"take-000-ir.json","timing_limit_hz":1250.0,"low_hz":20.0,"high_hz":20000.0}"#,
        )
        .unwrap();
        assert!(legacy.impulse_csv_file.is_none());
    }

    #[test]
    fn emitted_coherence_sidecar_loads_into_autoeq_curve() {
        // Cross-repo handshake: the exact files `write_phase` emits load
        // through autoeq's measurement reader with `Curve.coherence`
        // populated from the sidecar (requirements-roomeq-ingestion §2).
        let mut plan: CaptureSessionPlan =
            serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json"))
                .unwrap();
        plan.sweep.duration_secs = 0.25;
        let stimulus =
            super::super::protocol::prepare_capture_stimulus(&plan.clone().validate().unwrap())
                .unwrap();
        let output = tempfile::tempdir().unwrap();
        let flat = output.path().join("flat.txt");
        std::fs::write(&flat, "10 0\n24000 0\n").unwrap();
        let mut recorded = vec![0.0; stimulus.samples.len()];
        recorded[37..].copy_from_slice(&stimulus.samples[..stimulus.samples.len() - 37]);
        let artifacts =
            write_phase(&recorded, &stimulus, &plan, &flat, output.path(), 4, 20.0).unwrap();
        assert!(artifacts.coherence_file.is_some());
        let curve =
            autoeq::read::read_curve_from_csv(&output.path().join(&artifacts.response_file))
                .unwrap();
        let coherence = curve.coherence.expect("autoeq joined the sidecar");
        assert_eq!(coherence.len(), curve.freq.len());
        // A clean delayed copy coheres almost perfectly on every bin.
        let mean = coherence.iter().sum::<f64>() / coherence.len() as f64;
        assert!(mean > 0.95, "{mean}");
    }

    #[test]
    fn take_sti_orders_dry_above_reverberant_and_rejects_unsuitable() {
        let rate = 48000u32;
        let mut dry = vec![0.0; 48000];
        dry[100] = 1.0;
        let clean = take_sti(&dry, rate).unwrap();
        assert!(clean.sti > 0.99, "{}", clean.sti);
        assert_eq!(clean.mti.len(), 7);
        assert!((clean.duration_s - 1.0).abs() < 1e-9);
        // Exponentially decaying uniform noise: a miniature room IR with
        // genuine broadband energy in every octave band.
        let decay = |rt60: f64| -> Vec<f64> {
            let mut state = 0x243F_6A88_85A3_0B5Du64;
            (0..48000)
                .map(|n| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let noise = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
                    (-6.0 * 2f64.ln() * f64::from(n) / (rt60 * f64::from(rate))).exp() * noise
                })
                .collect()
        };
        let short = take_sti(&decay(0.5), rate).unwrap();
        let long = take_sti(&decay(2.0), rate).unwrap();
        assert!(
            clean.sti > short.sti && short.sti > long.sti,
            "{} {} {}",
            clean.sti,
            short.sti,
            long.sti
        );
        assert!(take_sti(&[], rate).is_none());
        assert!(take_sti(&vec![0.0; 48000], rate).is_none());
        assert!(take_sti(&vec![f64::NAN; 48000], rate).is_none());
        assert!(take_sti(&dry, 16000).is_none());
    }

    #[test]
    fn exported_ir_retains_delay_and_calibration_retains_phase() {
        let mut plan: CaptureSessionPlan =
            serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json"))
                .unwrap();
        plan.sweep.duration_secs = 0.25;
        let stimulus =
            super::super::protocol::prepare_capture_stimulus(&plan.clone().validate().unwrap())
                .unwrap();
        let output = tempfile::tempdir().unwrap();
        let flat = output.path().join("flat.txt");
        let raised = output.path().join("raised.txt");
        std::fs::write(&flat, "10 0\n24000 0\n").unwrap();
        std::fs::write(&raised, "10 6\n24000 6\n").unwrap();
        for delay in [37, 61] {
            let mut recorded = vec![0.0; stimulus.samples.len()];
            recorded[delay..].copy_from_slice(&stimulus.samples[..stimulus.samples.len() - delay]);
            let original =
                write_phase(&recorded, &stimulus, &plan, &flat, output.path(), 0, 20.0).unwrap();
            let calibrated =
                write_phase(&recorded, &stimulus, &plan, &raised, output.path(), 1, 20.0).unwrap();
            assert_eq!(original.timing_limit_hz, 1250.0);
            let ir: serde_json::Value = serde_json::from_slice(
                &std::fs::read(output.path().join(&original.impulse_file)).unwrap(),
            )
            .unwrap();
            let values = ir["samples"].as_array().unwrap();
            let peak = values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.as_f64()
                        .unwrap()
                        .abs()
                        .total_cmp(&b.as_f64().unwrap().abs())
                })
                .unwrap()
                .0;
            assert_eq!(peak, delay);
            let first =
                std::fs::read_to_string(output.path().join(original.response_file)).unwrap();
            let second =
                std::fs::read_to_string(output.path().join(calibrated.response_file)).unwrap();
            for (a, b) in first.lines().skip(1).zip(second.lines().skip(1)) {
                let a: Vec<f64> = a.split(',').map(|value| value.parse().unwrap()).collect();
                let b: Vec<f64> = b.split(',').map(|value| value.parse().unwrap()).collect();
                assert!((a[1] - b[1] - 6.0).abs() < 1e-8);
                assert!((a[2] - b[2]).abs() < 1e-8);
                let expected = -360.0 * a[0] * delay as f64 / f64::from(plan.sample_rate_hz);
                let error = (a[2] - expected + 180.0).rem_euclid(360.0) - 180.0;
                assert!(error.abs() < 1e-6, "phase error {error} at {} Hz", a[0]);
            }
        }
    }
}
