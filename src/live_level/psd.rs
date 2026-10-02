//! One-sided Hann power spectral density for bounded live analysis.
//!
//! Bin values are FS²/Hz. The cells assign each bin's full one-sided energy
//! `PSD * Fs / N` uniformly over its supported frequency interval. The DC and
//! Nyquist cells cover half a bin each, while retaining their full bin energy.

// Rust guideline compliant 2026-02-21

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Reuses one planned transform and its buffers for fixed-size live frames.
pub(super) struct PsdAnalyzer {
    fft_size: usize,
    transform: Arc<dyn Fft<f64>>,
    input: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    window: Vec<f64>,
    window_energy: f64,
    frequencies_hz: Vec<f64>,
    psd_fs2_per_hz: Vec<f64>,
}

impl PsdAnalyzer {
    /// Plans a forward FFT and precomputes a symmetric Hann window.
    pub(super) fn new(fft_size: usize) -> Result<Self, &'static str> {
        if fft_size < 4 || !fft_size.is_multiple_of(2) {
            return Err("PSD analysis requires an even FFT size of at least four");
        }
        let mut planner = FftPlanner::<f64>::new();
        let transform = planner.plan_fft_forward(fft_size);
        let scratch = vec![Complex::new(0.0, 0.0); transform.get_inplace_scratch_len()];
        let denominator = (fft_size - 1) as f64;
        let window: Vec<_> = (0..fft_size)
            .map(|index| 0.5 * (1.0 - (std::f64::consts::TAU * index as f64 / denominator).cos()))
            .collect();
        let window_energy = window.iter().map(|value| value * value).sum::<f64>();
        if !window_energy.is_finite() || window_energy <= 0.0 {
            return Err("PSD Hann window has invalid energy");
        }
        let bin_count = fft_size / 2 + 1;
        Ok(Self {
            fft_size,
            transform,
            input: vec![Complex::new(0.0, 0.0); fft_size],
            scratch,
            window,
            window_energy,
            frequencies_hz: vec![0.0; bin_count],
            psd_fs2_per_hz: vec![0.0; bin_count],
        })
    }

    /// Computes a one-sided PSD without retaining data between frames.
    pub(super) fn analyze(
        &mut self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<(), &'static str> {
        if samples.len() != self.fft_size {
            return Err("PSD frame length does not match its planned FFT size");
        }
        if sample_rate_hz == 0 {
            return Err("PSD sample rate must be positive");
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err("PSD input contains nonfinite samples");
        }

        for ((input, sample), window) in self.input.iter_mut().zip(samples).zip(&self.window) {
            *input = Complex::new(f64::from(*sample) * window, 0.0);
        }
        self.transform
            .process_with_scratch(&mut self.input, &mut self.scratch);

        let bin_width_hz = f64::from(sample_rate_hz) / self.fft_size as f64;
        let nyquist_bin = self.fft_size / 2;
        for bin in 0..=nyquist_bin {
            let endpoint = bin == 0 || bin == nyquist_bin;
            let one_sided_factor = if endpoint { 1.0 } else { 2.0 };
            let power = self.input[bin].norm_sqr() * one_sided_factor
                / (f64::from(sample_rate_hz) * self.window_energy);
            if !power.is_finite() || power < 0.0 {
                return Err("PSD transform produced an invalid bin");
            }
            self.frequencies_hz[bin] = bin as f64 * bin_width_hz;
            self.psd_fs2_per_hz[bin] = power;
        }
        Ok(())
    }

    /// Returns the inclusive DC-through-Nyquist frequency grid.
    pub(super) fn frequencies_hz(&self) -> &[f64] {
        &self.frequencies_hz
    }

    /// Returns the one-sided power spectral density in FS²/Hz.
    pub(super) fn psd_fs2_per_hz(&self) -> &[f64] {
        &self.psd_fs2_per_hz
    }
}

/// Integrates one-sided bin energy over a band, refusing any unsupported bin.
pub(super) fn integrate_band_power(
    psd: &[Option<f64>],
    fft_size: usize,
    sample_rate_hz: u32,
    band_hz: [f64; 2],
) -> Option<f64> {
    if fft_size < 4
        || !fft_size.is_multiple_of(2)
        || sample_rate_hz == 0
        || psd.len() != fft_size / 2 + 1
        || !band_hz[0].is_finite()
        || !band_hz[1].is_finite()
        || band_hz[0] < 0.0
        || band_hz[1] <= band_hz[0]
    {
        return None;
    }

    let bin_width_hz = f64::from(sample_rate_hz) / fft_size as f64;
    let nyquist_hz = f64::from(sample_rate_hz) / 2.0;
    if band_hz[1] > nyquist_hz {
        return None;
    }

    let nyquist_bin = fft_size / 2;
    let mut integrated_power = 0.0;
    for (bin, value) in psd.iter().enumerate() {
        let (cell_low_hz, cell_high_hz) = if bin == 0 {
            (0.0, bin_width_hz / 2.0)
        } else if bin == nyquist_bin {
            (nyquist_hz - bin_width_hz / 2.0, nyquist_hz)
        } else {
            (
                (bin as f64 - 0.5) * bin_width_hz,
                (bin as f64 + 0.5) * bin_width_hz,
            )
        };
        let overlap_hz = (band_hz[1].min(cell_high_hz) - band_hz[0].max(cell_low_hz)).max(0.0);
        if overlap_hz == 0.0 {
            continue;
        }
        let value = value.as_ref()?;
        if !value.is_finite() || *value < 0.0 {
            return None;
        }
        let cell_width_hz = cell_high_hz - cell_low_hz;
        if cell_width_hz <= 0.0 {
            return None;
        }
        // Endpoint cells have half the frequency width but retain a full PSD*df bin energy.
        integrated_power += *value * bin_width_hz * (overlap_hz / cell_width_hz);
    }
    integrated_power.is_finite().then_some(integrated_power)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_dft_psd(samples: &[f32], sample_rate_hz: u32) -> Vec<f64> {
        let size = samples.len();
        let window_energy: f64 = (0..size)
            .map(|index| {
                let window =
                    0.5 * (1.0 - (std::f64::consts::TAU * index as f64 / (size - 1) as f64).cos());
                window * window
            })
            .sum();
        (0..=size / 2)
            .map(|bin| {
                let real = (0..size)
                    .map(|index| {
                        let window = 0.5
                            * (1.0
                                - (std::f64::consts::TAU * index as f64 / (size - 1) as f64).cos());
                        f64::from(samples[index])
                            * window
                            * (std::f64::consts::TAU * bin as f64 * index as f64 / size as f64)
                                .cos()
                    })
                    .sum::<f64>();
                let imaginary = (0..size)
                    .map(|index| {
                        let window = 0.5
                            * (1.0
                                - (std::f64::consts::TAU * index as f64 / (size - 1) as f64).cos());
                        f64::from(samples[index])
                            * window
                            * (std::f64::consts::TAU * bin as f64 * index as f64 / size as f64)
                                .sin()
                    })
                    .sum::<f64>();
                let endpoint = bin == 0 || bin == size / 2;
                let factor = if endpoint { 1.0 } else { 2.0 };
                factor * (real * real + imaginary * imaginary)
                    / (f64::from(sample_rate_hz) * window_energy)
            })
            .collect()
    }

    #[test]
    fn planned_psd_matches_independent_direct_dft() {
        let size = 256;
        let sample_rate_hz = 16_000;
        let samples: Vec<_> = (0..size)
            .map(|index| {
                let index = index as f64;
                (0.21 * (std::f64::consts::TAU * 17.0 * index / size as f64).sin()
                    + 0.07 * (std::f64::consts::TAU * 41.0 * index / size as f64).cos()
                    + 0.001 * index) as f32
            })
            .collect();
        let expected = direct_dft_psd(&samples, sample_rate_hz);
        let mut analyzer = PsdAnalyzer::new(size).unwrap();
        analyzer.analyze(&samples, sample_rate_hz).unwrap();
        assert_eq!(analyzer.frequencies_hz().len(), size / 2 + 1);
        assert_eq!(analyzer.frequencies_hz()[0], 0.0);
        assert_eq!(
            analyzer.frequencies_hz()[size / 2],
            f64::from(sample_rate_hz) / 2.0
        );
        for (actual, expected) in analyzer.psd_fs2_per_hz().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn full_band_energy_matches_hann_weighted_mean_square_including_endpoints() {
        let size = 512;
        let sample_rate_hz = 48_000;
        let samples: Vec<_> = (0..size)
            .map(|index| {
                (0.3 * (std::f64::consts::TAU * 11.0 * index as f64 / size as f64).sin()
                    + 0.1 * (std::f64::consts::TAU * 31.0 * index as f64 / size as f64).cos())
                    as f32
            })
            .collect();
        let mut analyzer = PsdAnalyzer::new(size).unwrap();
        analyzer.analyze(&samples, sample_rate_hz).unwrap();
        let bin_width_hz = f64::from(sample_rate_hz) / size as f64;
        let bins: Vec<_> = analyzer
            .psd_fs2_per_hz()
            .iter()
            .copied()
            .map(Some)
            .collect();
        let integrated = integrate_band_power(
            &bins,
            size,
            sample_rate_hz,
            [0.0, f64::from(sample_rate_hz) / 2.0],
        )
        .unwrap();
        let windowed_mean_square = samples
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                let window =
                    0.5 * (1.0 - (std::f64::consts::TAU * index as f64 / (size - 1) as f64).cos());
                f64::from(*sample).powi(2) * window * window
            })
            .sum::<f64>()
            / (0..size)
                .map(|index| {
                    let window = 0.5
                        * (1.0 - (std::f64::consts::TAU * index as f64 / (size - 1) as f64).cos());
                    window * window
                })
                .sum::<f64>();
        assert!((integrated - windowed_mean_square).abs() < 1e-12);
        assert!((integrated - bins.iter().flatten().sum::<f64>() * bin_width_hz).abs() < 1e-12);
    }

    #[test]
    fn a_band_with_any_unsupported_overlapping_bin_is_unknown() {
        let bins = vec![Some(1.0), Some(1.0), None, Some(1.0), Some(1.0)];
        assert!(integrate_band_power(&bins, 8, 8_000, [0.0, 4_000.0]).is_none());
        assert!(integrate_band_power(&bins, 8, 8_000, [0.0, 400.0]).is_some());
    }
}
