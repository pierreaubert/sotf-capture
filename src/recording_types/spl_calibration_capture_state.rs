use super::spl_calibration_capture_status::SplCalibrationCaptureStatus;
pub use crate::signal_recorder::SplCalibrationResult;
pub use autoeq::roomeq::SplCalibration;

/// Shared business state for the SplCalibration step.
///
/// Collected from the user across two stages:
/// 1. Engine plays a reference tone and returns a `SplCalibrationResult`
///    (peak + RMS sample levels on the mic).
/// 2. User types the dBSPL their external meter reads while the tone
///    plays; that becomes `reported_db_spl`. `spl_offset_db` is derived
///    as `reported_db_spl − 20 · log10(peak_sample_level)`, matching
///    [`SplCalibration::dbspl_for_peak_level`].
///
/// On save, this state is converted to the `SplCalibration` struct
/// the autoeq `RecordingConfiguration` carries.
/// Persisted offsets have no basis marker, so old RMS-derived values cannot be
/// identified or migrated safely and require explicit recalibration.
#[derive(Debug, Clone)]
pub struct SplCalibrationCaptureState {
    /// Reference tone frequency (Hz). Default 1000.
    pub reference_freq_hz: f32,
    /// Digital amplitude of the reference tone (0.0..=1.0). Default 0.25
    /// — leaves ~12 dB headroom and reliably hits 75-85 dBSPL on typical
    /// home systems at normal volume.
    pub tone_amp: f32,
    /// Tone duration in seconds. Default 3.0.
    pub duration_s: f32,
    /// Sample rate used for the capture (Hz).
    pub sample_rate: u32,
    /// Playback output channel (0-based). Default 0 (left / mono).
    pub output_channel: u16,
    /// Microphone input channel (0-based).
    pub input_channel: u16,
    /// Capture status.
    pub status: SplCalibrationCaptureStatus,
    /// Raw engine capture result — `None` until a successful run.
    pub engine_result: Option<SplCalibrationResult>,
    /// dBSPL the user read from their external meter. `None` until
    /// the user has entered a value. Combines with
    /// `engine_result.peak_sample_level` to compute `spl_offset_db`.
    pub reported_db_spl: Option<f32>,
    /// Monotonic generation for stale-completion protection in both UIs.
    pub capture_generation: u64,
}

impl Default for SplCalibrationCaptureState {
    fn default() -> Self {
        Self {
            reference_freq_hz: 1000.0,
            tone_amp: 0.25,
            duration_s: 3.0,
            sample_rate: 48_000,
            output_channel: 0,
            input_channel: 0,
            status: SplCalibrationCaptureStatus::Idle,
            engine_result: None,
            reported_db_spl: None,
            capture_generation: 0,
        }
    }
}

impl SplCalibrationCaptureState {
    /// Invalidate prior SPL completions and return the new task generation.
    pub fn next_capture_generation(&mut self) -> u64 {
        self.capture_generation += 1;
        self.engine_result = None;
        self.reported_db_spl = None;
        self.status = SplCalibrationCaptureStatus::Idle;
        self.capture_generation
    }

    /// Return whether a completion belongs to the current SPL task.
    pub fn is_current_capture(&self, generation: u64) -> bool {
        self.capture_generation == generation
    }

    pub fn apply_engine_result(&mut self, result: SplCalibrationResult) {
        match result.validate() {
            Ok(()) => {
                self.engine_result = Some(result);
                self.status = SplCalibrationCaptureStatus::Complete;
            }
            Err(error) => {
                self.engine_result = None;
                self.status = SplCalibrationCaptureStatus::Failed(error);
            }
        }
    }

    /// `true` once the engine has captured a tone AND the user has
    /// typed the dBSPL their meter read. Consumers gate the Save /
    /// Continue action on this.
    pub fn is_ready(&self) -> bool {
        self.to_spl_calibration().is_some()
    }

    /// Derive the final `SplCalibration` once both the engine capture
    /// and the user-entered meter reading are present.
    pub fn to_spl_calibration(&self) -> Option<SplCalibration> {
        if !matches!(self.status, SplCalibrationCaptureStatus::Complete) {
            return None;
        }
        let er = self.engine_result.as_ref()?;
        er.validate().ok()?;
        let reported = self.reported_db_spl.filter(|reading| reading.is_finite())?;
        // SplCalibration maps peak sample values to SPL; keep the stored
        // anchor consistent with that shared peak-based conversion.
        let spl_offset_db = reported - 20.0 * er.peak_sample_level.log10();
        if !spl_offset_db.is_finite() {
            return None;
        }
        Some(SplCalibration {
            reported_db_spl: reported,
            reference_freq_hz: er.reference_freq_hz,
            peak_sample_level: er.peak_sample_level,
            spl_offset_db,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn captured() -> SplCalibrationResult {
        SplCalibrationResult {
            sample_rate: 48_000,
            peak_sample_level: 0.5,
            rms_sample_level: 0.25,
            reference_freq_hz: 1_000.0,
            output_channel: 0,
        }
    }

    fn state_from_samples(samples: &[f32]) -> SplCalibrationCaptureState {
        let peak_sample_level = samples
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0, f32::max);
        let rms_sample_level = (samples
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt() as f32;
        let mut state = SplCalibrationCaptureState {
            reported_db_spl: Some(75.0),
            ..SplCalibrationCaptureState::default()
        };
        state.apply_engine_result(SplCalibrationResult {
            sample_rate: 48_000,
            peak_sample_level,
            rms_sample_level,
            reference_freq_hz: 1_000.0,
            output_channel: 0,
        });
        state
    }

    fn assert_reference_peak_roundtrips(state: &SplCalibrationCaptureState) {
        let reference = state.engine_result.as_ref().unwrap();
        let anchor = state.to_spl_calibration().unwrap();
        assert!((anchor.dbspl_for_peak_level(reference.peak_sample_level) - 75.0).abs() < 1e-4);
        assert!((anchor.peak_level_for_dbspl(75.0) - reference.peak_sample_level).abs() < 1e-5);
        assert!(
            (anchor.spl_offset_db - (75.0 - 20.0 * reference.peak_sample_level.log10())).abs()
                < 1e-4
        );
    }

    #[test]
    fn calibration_requires_completed_capture_and_finite_meter_reading() {
        let mut state = SplCalibrationCaptureState::default();
        state.apply_engine_result(captured());
        assert!(!state.is_ready());
        for reading in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            state.reported_db_spl = Some(reading);
            assert!(!state.is_ready());
            assert!(state.to_spl_calibration().is_none());
        }
        state.reported_db_spl = Some(75.0);
        let anchor = state.to_spl_calibration().unwrap();
        assert!((anchor.spl_offset_db - 81.0206).abs() < 1e-4);
        assert!((anchor.dbspl_for_peak_level(anchor.peak_sample_level) - 75.0).abs() < 1e-4);
        // A later recording at twice the reference peak has a 6.0206 dB higher level.
        assert!((anchor.dbspl_for_peak_level(1.0) - 81.0206).abs() < 1e-4);
        state.status = SplCalibrationCaptureStatus::Running { started_at_ms: 0 };
        assert!(state.to_spl_calibration().is_none());
        state.next_capture_generation();
        assert!(state.engine_result.is_none());
        assert!(state.reported_db_spl.is_none());
        state.apply_engine_result(captured());
        assert!(
            !state.is_ready(),
            "a new capture requires its own meter reading"
        );
    }

    #[test]
    fn sine_reference_peak_roundtrips_reported_spl() {
        let samples: Vec<f32> = (0..48 * 8)
            .map(|sample| {
                let phase = std::f64::consts::TAU * 1_000.0 * sample as f64 / 48_000.0;
                (0.5 * phase.sin()) as f32
            })
            .collect();
        let state = state_from_samples(&samples);
        let result = state.engine_result.as_ref().unwrap();
        assert!(
            (result.rms_sample_level / result.peak_sample_level - 1.0 / 2.0_f32.sqrt()).abs()
                < 1e-4
        );
        assert_reference_peak_roundtrips(&state);
    }

    #[test]
    fn nonsinusoidal_reference_peak_roundtrips_its_distinct_crest_factor() {
        // A zero-mean 1 kHz pulse train with a 1/6 nonzero duty cycle.
        let samples: Vec<f32> = (0..48 * 8)
            .map(|sample| match sample % 48 {
                0..4 => 0.8,
                4..8 => -0.8,
                _ => 0.0,
            })
            .collect();
        let state = state_from_samples(&samples);
        let result = state.engine_result.as_ref().unwrap();
        assert!(
            (result.rms_sample_level / result.peak_sample_level - 1.0 / 6.0_f32.sqrt()).abs()
                < 1e-4
        );
        assert_reference_peak_roundtrips(&state);
    }

    #[test]
    fn invalid_capture_cannot_be_saved_or_marked_complete() {
        let valid = captured();
        let invalid = [
            SplCalibrationResult {
                rms_sample_level: 0.0,
                ..valid.clone()
            },
            SplCalibrationResult {
                rms_sample_level: -0.25,
                ..valid.clone()
            },
            SplCalibrationResult {
                rms_sample_level: f32::NAN,
                ..valid.clone()
            },
            SplCalibrationResult {
                peak_sample_level: f32::NAN,
                ..valid.clone()
            },
            SplCalibrationResult {
                peak_sample_level: f32::INFINITY,
                ..valid.clone()
            },
            SplCalibrationResult {
                peak_sample_level: 1.0,
                ..valid.clone()
            },
            SplCalibrationResult {
                rms_sample_level: 0.6,
                ..valid.clone()
            },
            SplCalibrationResult {
                sample_rate: 0,
                ..valid.clone()
            },
            SplCalibrationResult {
                reference_freq_hz: 24_000.0,
                ..valid.clone()
            },
            SplCalibrationResult {
                reference_freq_hz: f32::INFINITY,
                ..valid
            },
        ];
        for result in invalid {
            let mut state = SplCalibrationCaptureState {
                reported_db_spl: Some(75.0),
                ..SplCalibrationCaptureState::default()
            };
            state.apply_engine_result(result.clone());
            assert!(matches!(
                state.status,
                SplCalibrationCaptureStatus::Failed(_)
            ));
            assert!(state.engine_result.is_none());
            assert!(!state.is_ready());
            // Directly imported state must pass the same gate as engine completion.
            state.engine_result = Some(result);
            state.status = SplCalibrationCaptureStatus::Complete;
            assert!(state.to_spl_calibration().is_none());
        }
    }
}
