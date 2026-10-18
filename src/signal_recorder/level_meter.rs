//! Lock-free live input level meter fed from the capture callback.
//!
//! The meter is updated with plain atomic stores from the audio
//! callback (no allocation, no locking, no logging) and read from the
//! UI thread through [`CaptureLevelMeter::snapshot`]. Levels are raw
//! digital levels in dBFS; absolute SPL is only known once an SPL
//! calibration exists, so the UI labels the bar in dBFS and converts
//! with the stored calibration offset when one is present.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Headroom zone of the loudest recent block, classical meter colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelZone {
    /// No signal above −50 dBFS yet.
    Silence,
    /// Signal present but far below full scale (peak < −24 dBFS).
    Low,
    /// Healthy level with headroom (−24…−6 dBFS peak).
    Good,
    /// Close to full scale (−6…−1 dBFS peak): little headroom left.
    Hot,
    /// At or above −1 dBFS peak, or clipped samples observed.
    Clipping,
}

/// Post-capture microphone-gain advice derived from a finished take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GainAdvice {
    /// Clipping / no headroom occurred: lower the microphone gain.
    LowerMicrophoneGain,
    /// The take never rose above −30 dBFS peak: raise the gain.
    RaiseMicrophoneGain,
}

/// Immutable point-in-time view of a [`CaptureLevelMeter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptureLevelSnapshot {
    /// RMS of the most recent block, `None` before the first block.
    pub rms_dbfs: Option<f32>,
    /// Peak of the most recent block.
    pub peak_dbfs: Option<f32>,
    /// Loudest peak observed since reset (peak hold for the take).
    pub peak_hold_dbfs: Option<f32>,
    /// Loudest block RMS observed since reset.
    pub max_rms_dbfs: Option<f32>,
    /// Samples at or beyond digital full scale (or non-finite).
    pub clipped_samples: u64,
    /// Zone of the most recent block.
    pub zone: LevelZone,
}

impl CaptureLevelSnapshot {
    /// Zone thresholds shared by the live bar and the final verdict.
    #[must_use]
    pub fn zone_for(peak_dbfs: Option<f32>, clipped_samples: u64) -> LevelZone {
        if clipped_samples > 0 {
            return LevelZone::Clipping;
        }
        match peak_dbfs {
            Some(peak) if peak >= -1.0 => LevelZone::Clipping,
            Some(peak) if peak >= -6.0 => LevelZone::Hot,
            Some(peak) if peak >= -24.0 => LevelZone::Good,
            Some(peak) if peak >= -50.0 => LevelZone::Low,
            _ => LevelZone::Silence,
        }
    }

    /// Whether any block reached the red zone during the take.
    #[must_use]
    pub fn saw_red(&self) -> bool {
        self.clipped_samples > 0 || self.peak_hold_dbfs.is_some_and(|peak| peak >= -1.0)
    }

    /// Gain advice for a finished take; `None` when the level was usable.
    #[must_use]
    pub fn gain_advice(&self) -> Option<GainAdvice> {
        if self.saw_red() {
            return Some(GainAdvice::LowerMicrophoneGain);
        }
        match self.peak_hold_dbfs {
            Some(peak) if peak < -30.0 => Some(GainAdvice::RaiseMicrophoneGain),
            _ => None,
        }
    }
}

/// Shared live level state between the capture callback and the UI.
///
/// All fields are atomics; `record_*` methods are safe to call from a
/// realtime audio callback.
#[derive(Debug)]
pub struct CaptureLevelMeter {
    rms_dbfs_bits: AtomicU32,
    peak_dbfs_bits: AtomicU32,
    peak_hold_dbfs_bits: AtomicU32,
    max_rms_dbfs_bits: AtomicU32,
    clipped_samples: AtomicU64,
    blocks: AtomicU64,
}

const NONE_BITS: u32 = u32::MAX;

fn encode(level_dbfs: f32) -> u32 {
    if level_dbfs.is_finite() {
        level_dbfs.to_bits()
    } else {
        NONE_BITS
    }
}

fn decode(bits: u32) -> Option<f32> {
    if bits == NONE_BITS {
        None
    } else {
        Some(f32::from_bits(bits))
    }
}

fn store_max(bits: &AtomicU32, value_dbfs: f32) {
    let mut current = bits.load(Ordering::Relaxed);
    loop {
        let current_value = decode(current).unwrap_or(f32::NEG_INFINITY);
        if value_dbfs <= current_value {
            return;
        }
        match bits.compare_exchange_weak(
            current,
            encode(value_dbfs),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

impl Default for CaptureLevelMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptureLevelMeter {
    /// Create an empty meter (no block observed yet).
    #[must_use]
    pub fn new() -> Self {
        Self {
            rms_dbfs_bits: AtomicU32::new(NONE_BITS),
            peak_dbfs_bits: AtomicU32::new(NONE_BITS),
            peak_hold_dbfs_bits: AtomicU32::new(NONE_BITS),
            max_rms_dbfs_bits: AtomicU32::new(NONE_BITS),
            clipped_samples: AtomicU64::new(0),
            blocks: AtomicU64::new(0),
        }
    }

    /// Forget every observed block (start of a new take).
    pub fn reset(&self) {
        self.rms_dbfs_bits.store(NONE_BITS, Ordering::Relaxed);
        self.peak_dbfs_bits.store(NONE_BITS, Ordering::Relaxed);
        self.peak_hold_dbfs_bits.store(NONE_BITS, Ordering::Relaxed);
        self.max_rms_dbfs_bits.store(NONE_BITS, Ordering::Relaxed);
        self.clipped_samples.store(0, Ordering::Relaxed);
        self.blocks.store(0, Ordering::Relaxed);
    }

    /// Fold one mono block into the meter. Realtime-safe.
    pub fn record_mono(&self, samples: &[f32]) {
        self.record_interleaved(samples, 1, 0);
    }

    /// Fold one interleaved callback buffer, metering only `channel`.
    /// Realtime-safe: reads samples the callback already touches.
    pub fn record_interleaved(&self, data: &[f32], channels: usize, channel: usize) {
        if channels == 0 || channel >= channels || data.len() < channels {
            return;
        }
        let frames = data.len() / channels;
        self.record_values(
            frames,
            data.chunks_exact(channels).map(|frame| frame[channel]),
        );
    }

    #[cfg(feature = "capture-io")]
    pub(crate) fn record_converted<T: cpal::Sample>(
        &self,
        data: &[T],
        channels: usize,
        channel: usize,
    ) where
        f32: cpal::FromSample<T>,
    {
        if channels == 0 || channel >= channels || data.len() < channels {
            return;
        }
        self.record_values(
            data.len() / channels,
            data.chunks_exact(channels)
                .map(|frame| <f32 as cpal::FromSample<T>>::from_sample_(frame[channel])),
        );
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "Callback frame counts form an f64 RMS denominator; the meter deliberately publishes f32 levels."
    )]
    fn record_values(&self, frames: usize, samples: impl Iterator<Item = f32>) {
        let mut sum_sq = 0.0f64;
        let mut peak = 0.0f32;
        let mut clipped = 0u64;
        for sample in samples {
            if !sample.is_finite() {
                clipped += 1;
                continue;
            }
            let abs = sample.abs();
            if abs >= 0.999 {
                clipped += 1;
            }
            if abs > peak {
                peak = abs;
            }
            sum_sq += f64::from(sample) * f64::from(sample);
        }
        if clipped > 0 {
            self.clipped_samples.fetch_add(clipped, Ordering::Relaxed);
        }
        let rms = (sum_sq / frames as f64).sqrt() as f32;
        let rms_dbfs = if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            f32::NEG_INFINITY
        };
        let peak_dbfs = if peak > 0.0 {
            20.0 * peak.log10()
        } else {
            f32::NEG_INFINITY
        };
        self.rms_dbfs_bits
            .store(encode(rms_dbfs), Ordering::Relaxed);
        self.peak_dbfs_bits
            .store(encode(peak_dbfs), Ordering::Relaxed);
        store_max(&self.peak_hold_dbfs_bits, peak_dbfs);
        store_max(&self.max_rms_dbfs_bits, rms_dbfs);
        self.blocks.fetch_add(1, Ordering::Relaxed);
    }

    /// Read the current levels from a non-realtime thread.
    #[must_use]
    pub fn snapshot(&self) -> CaptureLevelSnapshot {
        let clipped_samples = self.clipped_samples.load(Ordering::Relaxed);
        let peak_dbfs = decode(self.peak_dbfs_bits.load(Ordering::Relaxed));
        CaptureLevelSnapshot {
            rms_dbfs: decode(self.rms_dbfs_bits.load(Ordering::Relaxed)),
            peak_dbfs,
            peak_hold_dbfs: decode(self.peak_hold_dbfs_bits.load(Ordering::Relaxed)),
            max_rms_dbfs: decode(self.max_rms_dbfs_bits.load(Ordering::Relaxed)),
            clipped_samples,
            zone: CaptureLevelSnapshot::zone_for(peak_dbfs, clipped_samples),
        }
    }

    /// Whether at least one block has been metered since reset.
    #[must_use]
    pub fn has_data(&self) -> bool {
        self.blocks.load(Ordering::Relaxed) > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_stays_in_silence_zone_without_advice() {
        let meter = CaptureLevelMeter::new();
        meter.record_mono(&[0.0; 512]);
        let snap = meter.snapshot();
        assert_eq!(snap.zone, LevelZone::Silence);
        assert_eq!(snap.peak_dbfs, None);
        assert_eq!(snap.peak_hold_dbfs, None);
        assert_eq!(snap.gain_advice(), None);
    }

    #[test]
    fn healthy_block_is_good_zone_without_advice() {
        let meter = CaptureLevelMeter::new();
        // −12 dBFS sine: RMS ≈ 0.25, peak ≈ 0.354.
        let samples: Vec<f32> = (0_u16..4800)
            .map(|i| 0.354 * (2.0 * std::f32::consts::PI * 1000.0 * f32::from(i) / 48_000.0).sin())
            .collect();
        meter.record_mono(&samples);
        let snap = meter.snapshot();
        assert_eq!(snap.zone, LevelZone::Good);
        assert!(snap.rms_dbfs.is_some_and(|rms| (rms - -12.0).abs() < 0.2));
        assert_eq!(snap.gain_advice(), None);
        assert!(!snap.saw_red());
    }

    #[test]
    fn clipping_block_is_red_and_advises_lower_gain() {
        let meter = CaptureLevelMeter::new();
        meter.record_mono(&[0.5, -1.0, 0.25, 1.0]);
        let snap = meter.snapshot();
        assert_eq!(snap.zone, LevelZone::Clipping);
        assert_eq!(snap.clipped_samples, 2);
        assert!(snap.saw_red());
        assert_eq!(snap.gain_advice(), Some(GainAdvice::LowerMicrophoneGain));
    }

    #[test]
    fn quiet_take_advises_higher_gain() {
        let meter = CaptureLevelMeter::new();
        // Peak −40 dBFS (0.01 linear): far too low for a measurement.
        meter.record_mono(&[0.01, -0.008, 0.005, -0.002]);
        let snap = meter.snapshot();
        assert_eq!(snap.zone, LevelZone::Low);
        assert_eq!(snap.gain_advice(), Some(GainAdvice::RaiseMicrophoneGain));
    }

    #[test]
    fn interleaved_meters_only_the_selected_channel() {
        let meter = CaptureLevelMeter::new();
        // Stereo: left loud (clipping), right quiet.
        let data = [1.0f32, 0.001, -1.0, -0.001, 0.5, 0.0005, -0.5, -0.0005];
        meter.record_interleaved(&data, 2, 1);
        let snap = meter.snapshot();
        assert_eq!(snap.clipped_samples, 0);
        assert!(snap.peak_dbfs.is_some_and(|peak| peak < -50.0));
    }

    #[test]
    fn reset_clears_peak_hold_and_clipping() {
        let meter = CaptureLevelMeter::new();
        meter.record_mono(&[1.0]);
        assert!(meter.snapshot().saw_red());
        meter.reset();
        let snap = meter.snapshot();
        assert_eq!(snap.zone, LevelZone::Silence);
        assert_eq!(snap.clipped_samples, 0);
        assert!(!meter.has_data());
    }
}
