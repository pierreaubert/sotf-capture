#[cfg(feature = "capture-io")]
use super::misc::actionable_capture_error;
#[cfg(feature = "capture-io")]
use super::types::MeasurementOutputConfig;
#[cfg(feature = "capture-io")]
use super::write::write_capture_pairs_to_ring;
#[cfg(feature = "capture-io")]
use cpal::Sample as _;
use math_audio_dsp::signals::gen_log_sweep_octave_scaled;
#[cfg(feature = "capture-io")]
use std::sync::Arc;
#[cfg(feature = "capture-io")]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// One mono signal and its routing schedule, independent of hardware width.
#[cfg(feature = "capture-io")]
#[derive(Debug)]
pub(super) struct SequentialPlayback {
    pub signal: Vec<f32>,
    pub routes: Vec<u16>,
    pub silence_frames: usize,
    pub channels: usize,
    pub total_frames: usize,
}

#[cfg(feature = "capture-io")]
impl SequentialPlayback {
    fn fill<T: cpal::Sample + cpal::FromSample<f32>>(&self, data: &mut [T], cursor: &AtomicUsize) {
        data.fill(T::from_sample(0.0));
        let first = cursor.load(Ordering::Relaxed);
        let frames = data.len() / self.channels;
        let segment = self.silence_frames + self.signal.len();
        for (index, frame) in data.chunks_exact_mut(self.channels).enumerate() {
            let position = first + index;
            if position < self.silence_frames || position >= self.total_frames {
                continue;
            }
            let relative = position - self.silence_frames;
            let route = relative / segment;
            let sample = relative % segment;
            if sample < self.signal.len() {
                frame[self.routes[route] as usize] =
                    T::from_sample(self.signal[sample].clamp(-1.0, 1.0));
            }
        }
        cursor.store(
            first.saturating_add(frames).min(self.total_frames),
            Ordering::Relaxed,
        );
    }
}

#[cfg(all(test, feature = "capture-io"))]
mod playback_tests {
    use super::*;

    #[test]
    fn schedule_matches_independent_dense_routing_across_callback_boundaries() {
        let playback = SequentialPlayback {
            signal: vec![0.1, -0.2, 0.3],
            routes: vec![2, 0],
            silence_frames: 2,
            channels: 4,
            total_frames: 12,
        };
        let mut expected = vec![[0.0; 4]; 15];
        expected[2][2] = 0.1;
        expected[3][2] = -0.2;
        expected[4][2] = 0.3;
        expected[7][0] = 0.1;
        expected[8][0] = -0.2;
        expected[9][0] = 0.3;
        let cursor = AtomicUsize::new(0);
        let mut actual = Vec::new();
        for frames in [1, 3, 2, 5, 4] {
            let mut buffer = vec![99.0_f32; frames * 4];
            playback.fill(&mut buffer, &cursor);
            actual.extend(buffer);
        }
        assert_eq!(actual, expected.into_iter().flatten().collect::<Vec<_>>());
        assert_eq!(cursor.load(Ordering::Relaxed), 12);
        assert_eq!(playback.signal.len() + playback.routes.len(), 5);
    }
}

/// Build an octave-scaled sweep surrounded by silence windows.
///
/// Returns `[pre_silence | sweep | post_silence]` as a flat `Vec<f32>`.
/// This is the core of the GD-Opt v2 Phase GD-1b signal path.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "Nonnegative silence durations round to whole frames using the same f32 rate as the sweep generator."
)]
pub(super) fn build_octave_sweep_with_silence(
    start_freq: f32,
    end_freq: f32,
    amp: f32,
    bass_octave_duration_s: f32,
    pre_silence_s: f32,
    post_silence_s: f32,
    sample_rate: u32,
) -> Vec<f32> {
    // Clamp bass duration per the design contract.
    let bass_dur = bass_octave_duration_s.clamp(1.0, 10.0);

    // The minimum total sweep duration is one full second so the upper octaves
    // remain usable even with a very narrow frequency range.
    let min_sweep_dur = 1.0_f32;

    let sweep = gen_log_sweep_octave_scaled(
        start_freq,
        end_freq,
        amp,
        sample_rate,
        bass_dur,
        min_sweep_dur,
    );

    let pre_n = (pre_silence_s.max(0.0) * sample_rate as f32).round() as usize;
    let post_n = (post_silence_s.max(0.0) * sample_rate as f32).round() as usize;

    let total = pre_n + sweep.len() + post_n;
    let mut out = vec![0.0_f32; total];
    out[pre_n..pre_n + sweep.len()].copy_from_slice(&sweep);
    out
}

#[cfg(feature = "capture-io")]
pub(super) fn build_measurement_output_stream(
    device: &cpal::Device,
    output_config: MeasurementOutputConfig,
    playback: Arc<SequentialPlayback>,
    cursor: Arc<std::sync::atomic::AtomicUsize>,
    failed: Arc<AtomicBool>,
    presentation: Arc<super::presentation::PresentationClock>,
    log_tag: &str,
) -> Result<cpal::Stream, String> {
    let config = cpal::StreamConfig {
        channels: output_config.channels,
        sample_rate: output_config.sample_rate,
        buffer_size: cpal::BufferSize::Default,
    };
    macro_rules! build_typed {
        ($sample_ty:ty) => {
            build_measurement_output_stream_typed::<$sample_ty>(
                device,
                &config,
                playback,
                cursor,
                failed,
                presentation,
                log_tag,
            )
        };
    }
    match output_config.sample_format {
        cpal::SampleFormat::F32 => build_typed!(f32),
        cpal::SampleFormat::F64 => build_typed!(f64),
        cpal::SampleFormat::I8 => build_typed!(i8),
        cpal::SampleFormat::I16 => build_typed!(i16),
        cpal::SampleFormat::I24 => build_typed!(cpal::I24),
        cpal::SampleFormat::I32 => build_typed!(i32),
        cpal::SampleFormat::I64 => build_typed!(i64),
        cpal::SampleFormat::U8 => build_typed!(u8),
        cpal::SampleFormat::U16 => build_typed!(u16),
        cpal::SampleFormat::U24 => build_typed!(cpal::U24),
        cpal::SampleFormat::U32 => build_typed!(u32),
        cpal::SampleFormat::U64 => build_typed!(u64),
        other => Err(format!(
            "[{log_tag}] Unsupported measurement output sample format: {other:?}"
        )),
    }
}

#[cfg(feature = "capture-io")]
pub(super) fn build_measurement_output_stream_typed<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    playback: Arc<SequentialPlayback>,
    cursor: Arc<std::sync::atomic::AtomicUsize>,
    failed: Arc<AtomicBool>,
    presentation: Arc<super::presentation::PresentationClock>,
    log_tag: &str,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    use cpal::traits::DeviceTrait;
    let data_failed = Arc::clone(&failed);
    let rate = config.sample_rate;

    device
        .build_output_stream(
            config,
            move |data: &mut [T], info| {
                let first = cursor.load(Ordering::Relaxed);
                if !data.len().is_multiple_of(playback.channels) {
                    data_failed.store(true, Ordering::Release);
                }
                playback.fill(data, &cursor);
                if first < playback.total_frames
                    && cursor.load(Ordering::Relaxed) >= playback.total_frames
                {
                    presentation.submit(info, playback.total_frames - first, rate, &data_failed);
                }
            },
            move |_| failed.store(true, Ordering::Release),
            None,
        )
        .map_err(|e| {
            actionable_capture_error(
                &format!("[{log_tag}] Failed to build {:?} output stream", T::FORMAT),
                &e,
            )
        })
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "capture-io")]
pub(super) fn build_measurement_input_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    capture_producer: rtrb::Producer<(f32, f32)>,
    capture_count_callback: Arc<AtomicUsize>,
    capture_overruns_callback: Arc<AtomicUsize>,
    failed: Arc<AtomicBool>,
    hw_ch: usize,
    input_ch_idx: usize,
    loopback_ch_idx: Option<usize>,
    log_tag: &str,
) -> Result<cpal::Stream, String> {
    macro_rules! build_typed {
        ($sample_ty:ty) => {
            build_measurement_input_stream_typed::<$sample_ty>(
                device,
                config,
                capture_producer,
                capture_count_callback,
                capture_overruns_callback,
                failed,
                hw_ch,
                input_ch_idx,
                loopback_ch_idx,
                log_tag,
            )
        };
    }
    match sample_format {
        cpal::SampleFormat::F32 => build_typed!(f32),
        cpal::SampleFormat::F64 => build_typed!(f64),
        cpal::SampleFormat::I8 => build_typed!(i8),
        cpal::SampleFormat::I16 => build_typed!(i16),
        cpal::SampleFormat::I24 => build_typed!(cpal::I24),
        cpal::SampleFormat::I32 => build_typed!(i32),
        cpal::SampleFormat::I64 => build_typed!(i64),
        cpal::SampleFormat::U8 => build_typed!(u8),
        cpal::SampleFormat::U16 => build_typed!(u16),
        cpal::SampleFormat::U24 => build_typed!(cpal::U24),
        cpal::SampleFormat::U32 => build_typed!(u32),
        cpal::SampleFormat::U64 => build_typed!(u64),
        other => Err(format!(
            "[{log_tag}] Unsupported measurement input sample format: {other:?}"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "capture-io")]
pub(super) fn build_measurement_input_stream_typed<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut capture_producer: rtrb::Producer<(f32, f32)>,
    capture_count_callback: Arc<AtomicUsize>,
    capture_overruns_callback: Arc<AtomicUsize>,
    failed: Arc<AtomicBool>,
    hw_ch: usize,
    input_ch_idx: usize,
    loopback_ch_idx: Option<usize>,
    log_tag: &str,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    use cpal::traits::DeviceTrait;

    let callback_failed = Arc::clone(&failed);
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                capture_pairs(
                    data,
                    hw_ch,
                    input_ch_idx,
                    loopback_ch_idx,
                    &mut capture_producer,
                    InputCounters {
                        accepted: &capture_count_callback,
                        lost: &capture_overruns_callback,
                        failed: &callback_failed,
                    },
                );
            },
            move |_| failed.store(true, Ordering::Release),
            None,
        )
        .map_err(|e| {
            actionable_capture_error(
                &format!("[{log_tag}] Failed to build {:?} input stream", T::FORMAT),
                &e,
            )
        })
}

#[cfg(feature = "capture-io")]
#[derive(Debug, Clone, Copy)]
struct InputCounters<'a> {
    accepted: &'a AtomicUsize,
    lost: &'a AtomicUsize,
    failed: &'a AtomicBool,
}

#[cfg(feature = "capture-io")]
fn capture_pairs<T: cpal::Sample>(
    data: &[T],
    channels: usize,
    input: usize,
    loopback: Option<usize>,
    producer: &mut rtrb::Producer<(f32, f32)>,
    counters: InputCounters<'_>,
) where
    f32: cpal::FromSample<T>,
{
    if channels == 0
        || input >= channels
        || loopback.is_some_and(|channel| channel >= channels)
        || !data.len().is_multiple_of(channels)
        || data.chunks_exact(channels).any(|frame| {
            !f32::from_sample(frame[input]).is_finite()
                || loopback.is_some_and(|channel| !f32::from_sample(frame[channel]).is_finite())
        })
    {
        counters.failed.store(true, Ordering::Release);
        return;
    }
    let frames = data.len() / channels;
    let written = write_capture_pairs_to_ring(producer, data, channels, input, loopback);
    counters.accepted.fetch_add(written, Ordering::Relaxed);
    if written < frames {
        counters.lost.fetch_add(frames - written, Ordering::Relaxed);
    }
}

#[cfg(all(test, feature = "capture-io"))]
mod input_tests {
    use super::*;

    #[test]
    fn overflow_counts_missing_frames_and_retains_the_actual_discontinuity() {
        let (mut producer, mut consumer) = rtrb::RingBuffer::new(2);
        let accepted = AtomicUsize::new(0);
        let lost = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let counters = InputCounters {
            accepted: &accepted,
            lost: &lost,
            failed: &failed,
        };
        capture_pairs(
            &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
            2,
            0,
            Some(1),
            &mut producer,
            counters,
        );
        assert_eq!(consumer.pop().unwrap(), (1.0, 10.0));
        assert_eq!(consumer.pop().unwrap(), (2.0, 20.0));
        assert_eq!(lost.load(Ordering::Relaxed), 2);
        capture_pairs(
            &[5.0, 50.0, 6.0, 60.0],
            2,
            0,
            Some(1),
            &mut producer,
            counters,
        );
        assert_eq!(consumer.pop().unwrap(), (5.0, 50.0));
        assert_eq!(consumer.pop().unwrap(), (6.0, 60.0));
        assert_eq!(accepted.load(Ordering::Relaxed), 4);
        assert_eq!(lost.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn malformed_or_nonfinite_selected_input_latches_failure() {
        for samples in [vec![0.1_f32], vec![f32::NAN, 0.0], vec![0.0, f32::INFINITY]] {
            let (mut producer, mut consumer) = rtrb::RingBuffer::new(8);
            let accepted = AtomicUsize::new(0);
            let lost = AtomicUsize::new(0);
            let failed = AtomicBool::new(false);
            capture_pairs(
                &samples,
                2,
                0,
                Some(1),
                &mut producer,
                InputCounters {
                    accepted: &accepted,
                    lost: &lost,
                    failed: &failed,
                },
            );
            assert!(failed.load(Ordering::Acquire));
            assert_eq!(accepted.load(Ordering::Relaxed), 0);
            assert!(consumer.pop().is_err());
        }
    }
}
