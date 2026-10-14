// Rust guideline compliant 2026-02-21

//! Welch magnitude-squared coherence between two recordings.
//!
//! Coherence backs Group-Delay optimization downstream: GD weights each bin
//! by how linearly the recorded take follows its reference. [`spectral_coherence`]
//! estimates γ²(f) = |Gxy|²/(Gxx·Gyy) from Hann-windowed, half-overlapping
//! segments. Delays short against the segment length preserve coherence, so
//! the pair needs no alignment (take/reference acoustic delays are ~1 ms
//! against ~1 s segments); bins without signal energy report 0, not NaN.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use std::f64::consts::PI;

/// Welch segment count for coherence estimation.
const COHERENCE_SEGMENTS: usize = 8;
/// Shortest usable Welch segment, in samples.
const MIN_SEGMENT_SAMPLES: usize = 256;

/// Estimate magnitude-squared coherence per nonnegative-frequency bin.
///
/// Returns `(frequency_hz, coherence)` pairs on the segment FFT grid.
/// Both inputs must share their length; acoustic-scale delays need no
/// alignment. Fails when the pair is empty, mismatched, or too short for
/// [`COHERENCE_SEGMENTS`] half-overlapping segments.
#[expect(
    clippy::cast_precision_loss,
    reason = "FFT bin indices and buffer lengths are converted to f64 for frequency coordinates."
)]
pub(super) fn spectral_coherence(
    recorded: &[f32],
    reference: &[f32],
    sample_rate_hz: u32,
) -> Result<Vec<(f64, f64)>, String> {
    if recorded.is_empty() || recorded.len() != reference.len() {
        return Err("coherence requires two nonempty recordings of equal length".into());
    }
    if sample_rate_hz == 0 {
        return Err("coherence requires a positive sample rate".into());
    }
    let segment_len = recorded.len() * 2 / (COHERENCE_SEGMENTS + 1);
    if segment_len < MIN_SEGMENT_SAMPLES {
        return Err("recording is too short for coherence estimation".into());
    }
    let step = segment_len / 2;
    let bins = segment_len / 2 + 1;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(segment_len);
    let mut cross = vec![Complex64::default(); bins];
    let mut auto_recorded = vec![0.0; bins];
    let mut auto_reference = vec![0.0; bins];
    let mut windowed_x = vec![Complex64::default(); segment_len];
    let mut windowed_y = vec![Complex64::default(); segment_len];
    for index in 0..COHERENCE_SEGMENTS {
        let start = index * step;
        fill_windowed(&recorded[start..start + segment_len], &mut windowed_x);
        fill_windowed(&reference[start..start + segment_len], &mut windowed_y);
        fft.process(&mut windowed_x);
        fft.process(&mut windowed_y);
        for bin in 0..bins {
            cross[bin] += windowed_x[bin] * windowed_y[bin].conj();
            auto_recorded[bin] += windowed_x[bin].norm_sqr();
            auto_reference[bin] += windowed_y[bin].norm_sqr();
        }
    }
    let rate = f64::from(sample_rate_hz);
    Ok((0..bins)
        .map(|bin| {
            let denominator = auto_recorded[bin] * auto_reference[bin];
            let coherence = if denominator > 0.0 {
                (cross[bin].norm_sqr() / denominator).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (bin as f64 * rate / segment_len as f64, coherence)
        })
        .collect())
}

/// Copy samples into a Hann-windowed complex buffer of equal length.
#[expect(
    clippy::cast_precision_loss,
    reason = "Hann-window coefficients use floating-point sample positions over an allocated audio buffer."
)]
fn fill_windowed(samples: &[f32], windowed: &mut [Complex64]) {
    let last = (samples.len() - 1) as f64;
    for (n, sample) in samples.iter().enumerate() {
        let window = 0.5 - 0.5 * (2.0 * PI * n as f64 / last).cos();
        windowed[n] = Complex64::new(f64::from(*sample) * window, 0.0);
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Synthetic fixtures use bounded sample counts and intentionally quantized audio; exact comparisons verify unchanged settings and samples."
)]
mod tests {
    use super::*;

    /// Deterministic uniform noise in [-1, 1] for coherence fixtures.
    fn noise(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    fn mean_coherence(grid: &[(f64, f64)]) -> f64 {
        grid.iter().map(|(_, gamma)| gamma).sum::<f64>() / grid.len() as f64
    }

    #[test]
    fn identical_signals_cohere_perfectly() {
        let signal = noise(48000, 7);
        let grid = spectral_coherence(&signal, &signal, 48000).unwrap();
        assert_eq!(grid.len(), 48000 * 2 / 9 / 2 + 1);
        assert!(grid.iter().all(|(_, gamma)| (gamma - 1.0).abs() < 1e-9));
        assert_eq!(grid[0].0, 0.0);
    }

    #[test]
    fn short_delay_preserves_coherence() {
        let reference = noise(48000, 11);
        let mut recorded = reference.clone();
        recorded.rotate_right(37);
        let grid = spectral_coherence(&recorded, &reference, 48000).unwrap();
        let mean = mean_coherence(&grid);
        assert!(mean > 0.999, "{mean}");
    }

    #[test]
    fn uncorrelated_noise_coheres_near_one_over_segments() {
        let grid = spectral_coherence(&noise(96000, 13), &noise(96000, 17), 48000).unwrap();
        let mean = mean_coherence(&grid);
        assert!(
            (0.02..0.30).contains(&mean),
            "uncorrelated mean coherence {mean} is outside the Welch expectation"
        );
    }

    #[test]
    fn added_noise_degrades_coherence_monotonically() {
        let reference = noise(96000, 19);
        let disturbance = noise(96000, 23);
        let mix = |amount: f32| -> Vec<f32> {
            reference
                .iter()
                .zip(&disturbance)
                .map(|(clean, dirt)| clean + amount * dirt)
                .collect()
        };
        let mild = mean_coherence(&spectral_coherence(&mix(0.25), &reference, 48000).unwrap());
        let heavy = mean_coherence(&spectral_coherence(&mix(1.0), &reference, 48000).unwrap());
        assert!(mild > 0.9, "{mild}");
        assert!(heavy > 0.4 && heavy < 0.65, "{heavy}");
    }

    #[test]
    fn silent_bins_report_zero_without_nan() {
        let zeros = vec![0.0; 48000];
        let grid = spectral_coherence(&zeros, &noise(48000, 29), 48000).unwrap();
        assert!(grid.iter().all(|(_, gamma)| *gamma == 0.0));
    }

    #[test]
    fn mismatched_or_short_pairs_are_refused() {
        assert!(spectral_coherence(&[], &[], 48000).is_err());
        assert!(spectral_coherence(&vec![0.1; 48000], &vec![0.1; 47999], 48000).is_err());
        assert!(spectral_coherence(&[0.1; 100], &[0.1; 100], 48000).is_err());
        assert!(spectral_coherence(&vec![0.1; 48000], &vec![0.1; 48000], 0).is_err());
    }
}
