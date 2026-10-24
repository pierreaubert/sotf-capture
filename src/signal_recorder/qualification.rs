//! Hardware-free probes of production callback cores and acquisition storage.
//!
//! Available only with `test-util` and `capture-io`. Probes exclude native driver
//! buffers, device scheduling, timestamp delivery, and subsequent DSP analysis.

// Rust guideline compliant 2026-02-21

use super::build::{InputCounters, SequentialPlayback, capture_pairs};
use super::multi_capture::{
    allocate_recordings, capture_frames_tracked, input_ring, playback_frames,
};
use super::recorder::{InputRings, capture_channels};
use super::types::CaptureOutputSegment;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

mod rejected;

/// Dimensions of one hardware-free callback case.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CallbackCase {
    /// Production callback core exercised by this case.
    pub core: &'static str,
    /// Native PCM representation.
    pub format: &'static str,
    /// Hardware channel width, including unselected channels.
    pub channels: usize,
    /// Number of input channels retained (one for output cases).
    pub selected: usize,
    /// Callback frame count.
    pub frames: usize,
}

/// Visit prepared callback cases with separate callback and control-thread work.
///
/// The visitor may repeatedly invoke the first closure, then the second closure
/// to drain input rings or reset completed playback. Time only the first closure.
/// Allocate instrumentation before invoking either closure. No devices are opened.
/// Covers every supported PCM type, ring overflow, malformed blocks, and late
/// non-finite float samples. Rejected-case expectations are checked after visiting.
///
/// # Panics
/// Panics if callback behavior contradicts a case's expectations or the visitor panics.
pub fn visit_callbacks(mut visitor: impl FnMut(CallbackCase, &mut dyn FnMut(), &mut dyn FnMut())) {
    for channels in [2, 8, 33, 94] {
        for frames in [32, 128, 512, 1024] {
            visit_typed::<f32>("f32", channels, frames, &mut visitor);
            visit_typed::<f64>("f64", channels, frames, &mut visitor);
            visit_typed::<i8>("i8", channels, frames, &mut visitor);
            visit_typed::<i16>("i16", channels, frames, &mut visitor);
            visit_typed::<cpal::I24>("i24", channels, frames, &mut visitor);
            visit_typed::<i32>("i32", channels, frames, &mut visitor);
            visit_typed::<i64>("i64", channels, frames, &mut visitor);
            visit_typed::<u8>("u8", channels, frames, &mut visitor);
            visit_typed::<u16>("u16", channels, frames, &mut visitor);
            visit_typed::<cpal::U24>("u24", channels, frames, &mut visitor);
            visit_typed::<u32>("u32", channels, frames, &mut visitor);
            visit_typed::<u64>("u64", channels, frames, &mut visitor);
            rejected::visit_nonfinite::<f32>("f32", channels, frames, &mut visitor);
            rejected::visit_nonfinite::<f64>("f64", channels, frames, &mut visitor);
        }
    }
}

fn visit_typed<T>(
    format: &'static str,
    channels: usize,
    frames: usize,
    visitor: &mut impl FnMut(CallbackCase, &mut dyn FnMut(), &mut dyn FnMut()),
) where
    T: cpal::SizedSample + cpal::FromSample<f32>,
    f32: cpal::FromSample<T>,
{
    let selected: Vec<_> = (0..channels.min(33))
        .map(|channel| u16::try_from(channel).expect("hardware width is at most 94"))
        .collect();
    let input = vec![T::from_sample_(0.25); channels * frames];
    let case = CallbackCase {
        core: "multidevice_input",
        format,
        channels,
        selected: selected.len(),
        frames,
    };
    visit_sweep_input(&input, case, visitor);
    visit_auxiliary_input(&input, case, visitor);
    rejected::visit(
        &input[..input.len() - 1],
        case,
        rejected::Fault::Malformed,
        visitor,
    );
    let (mut producer, mut consumer) = rtrb::RingBuffer::new(frames * selected.len());
    let failed = AtomicBool::new(false);
    let continuity = super::multi_capture::continuity::Continuity::default();
    visitor(
        case,
        &mut || {
            capture_frames_tracked(
                black_box(&input),
                channels,
                &selected,
                &mut producer,
                &failed,
                &continuity,
            );
        },
        &mut || {
            while let Ok(sample) = consumer.pop() {
                black_box(sample);
            }
        },
    );
    assert!(!failed.load(Ordering::Relaxed));

    while producer.push(0.25).is_ok() {}
    let overflowed = AtomicBool::new(false);
    visitor(
        CallbackCase {
            core: "multidevice_input_overflow",
            ..case
        },
        &mut || {
            capture_frames_tracked(
                black_box(&input),
                channels,
                &selected,
                &mut producer,
                &overflowed,
                &continuity,
            );
        },
        &mut || {},
    );
    assert!(overflowed.load(Ordering::Relaxed));

    let mut output = vec![T::from_sample_(0.0); channels * frames];
    let stimulus = vec![0.25; 1024];
    let regions: Vec<_> = (0..16)
        .map(|index| CaptureOutputSegment {
            start_frame: index * 64,
            end_frame: index * 64 + 32,
            channel: u16::try_from(index % channels).expect("bounded channel"),
        })
        .collect();
    let cursor = AtomicUsize::new(0);
    visitor(
        CallbackCase {
            core: "multidevice_output",
            selected: 1,
            ..case
        },
        &mut || {
            playback_frames(
                black_box(&mut output),
                black_box(channels),
                channels - 1,
                &regions,
                black_box(&stimulus),
                &cursor,
                &failed,
            );
            black_box(&output);
        },
        &mut || {
            if cursor.load(Ordering::Relaxed) >= stimulus.len() {
                cursor.store(0, Ordering::Relaxed);
            }
        },
    );
    assert!(!failed.load(Ordering::Relaxed));

    visit_auxiliary_output(case, &mut output, visitor);
}

fn visit_auxiliary_output<T>(
    case: CallbackCase,
    output: &mut [T],
    visitor: &mut impl FnMut(CallbackCase, &mut dyn FnMut(), &mut dyn FnMut()),
) where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let routes: Vec<_> = (0..case.channels.min(64))
        .map(|channel| u16::try_from(channel).expect("bounded channel"))
        .collect();
    let silence_frames = 16;
    let total_frames = (128 + silence_frames) * routes.len() + silence_frames;
    let playback = SequentialPlayback {
        signal: vec![0.25; 128],
        routes,
        silence_frames,
        channels: case.channels,
        total_frames,
    };
    let cursor = AtomicUsize::new(0);
    visitor(
        CallbackCase {
            core: "auxiliary_output",
            selected: 1,
            ..case
        },
        &mut || {
            black_box(&playback).fill(black_box(output), &cursor);
            black_box(&output);
        },
        &mut || {
            if cursor.load(Ordering::Relaxed) >= total_frames {
                cursor.store(0, Ordering::Relaxed);
            }
        },
    );
}

fn visit_sweep_input<T>(
    input: &[T],
    case: CallbackCase,
    visitor: &mut impl FnMut(CallbackCase, &mut dyn FnMut(), &mut dyn FnMut()),
) where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let (mut producers, mut consumers): (Vec<_>, Vec<_>) = (0..case.selected)
        .map(|_| rtrb::RingBuffer::new(case.frames))
        .unzip();
    let channels: Vec<_> = (0..case.selected).collect();
    let counts: Vec<_> = (0..case.selected)
        .map(|_| std::sync::Arc::new(AtomicUsize::new(0)))
        .collect();
    let failed = AtomicBool::new(false);
    let overruns = AtomicUsize::new(0);
    let meter = super::level_meter::CaptureLevelMeter::new();
    let continuity: Vec<_> = (0..case.selected)
        .map(|_| std::sync::Arc::new(super::continuity::Continuity::default()))
        .collect();
    visitor(
        CallbackCase {
            core: "sweep_input_with_meter",
            ..case
        },
        &mut || {
            capture_channels(
                black_box(input),
                case.channels,
                &mut InputRings {
                    producers: &mut producers,
                    counts: &counts,
                    channels: &channels,
                    overruns: &overruns,
                    failed: &failed,
                    meter: Some(&meter),
                    continuity: &continuity,
                },
            );
        },
        &mut || {
            for consumer in &mut consumers {
                while let Ok(sample) = consumer.pop() {
                    black_box(sample);
                }
            }
        },
    );
    assert!(!failed.load(Ordering::Relaxed));
    assert_eq!(overruns.load(Ordering::Relaxed), 0);
    for producer in &mut producers {
        while producer.push(0.25).is_ok() {}
    }
    visitor(
        CallbackCase {
            core: "sweep_input_overflow",
            ..case
        },
        &mut || {
            capture_channels(
                black_box(input),
                case.channels,
                &mut InputRings {
                    producers: &mut producers,
                    counts: &counts,
                    channels: &channels,
                    overruns: &overruns,
                    failed: &failed,
                    meter: Some(&meter),
                    continuity: &continuity,
                },
            );
        },
        &mut || {},
    );
    assert!(overruns.load(Ordering::Relaxed) > 0);
}

fn visit_auxiliary_input<T>(
    input: &[T],
    case: CallbackCase,
    visitor: &mut impl FnMut(CallbackCase, &mut dyn FnMut(), &mut dyn FnMut()),
) where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let (mut producer, mut consumer) = rtrb::RingBuffer::new(case.frames);
    let accepted = AtomicUsize::new(0);
    let lost = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let continuity = super::continuity::Continuity::default();
    visitor(
        CallbackCase {
            core: "auxiliary_input_with_loopback",
            selected: 2,
            ..case
        },
        &mut || {
            capture_pairs(
                black_box(input),
                case.channels,
                case.channels - 1,
                Some(0),
                &mut producer,
                InputCounters {
                    accepted: &accepted,
                    lost: &lost,
                    failed: &failed,
                    continuity: &continuity,
                },
            );
        },
        &mut || {
            while let Ok(pair) = consumer.pop() {
                black_box(pair);
            }
        },
    );
    assert!(!failed.load(Ordering::Relaxed));
    assert_eq!(lost.load(Ordering::Relaxed), 0);
    while producer.push((0.25, 0.25)).is_ok() {}
    visitor(
        CallbackCase {
            core: "auxiliary_input_overflow",
            selected: 2,
            ..case
        },
        &mut || {
            capture_pairs(
                black_box(input),
                case.channels,
                case.channels - 1,
                Some(0),
                &mut producer,
                InputCounters {
                    accepted: &accepted,
                    lost: &lost,
                    failed: &failed,
                    continuity: &continuity,
                },
            );
        },
        &mut || {},
    );
    assert!(lost.load(Ordering::Relaxed) > 0);
}

/// Payload dimensions for the acquisition storage probe.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct StorageCase {
    /// Selected microphone and loopback channels.
    pub inputs: usize,
    /// Retained frames per input, including startup and tail headroom.
    pub frames: usize,
    /// Nominal device rate.
    pub rate: u32,
    /// Frames in each of the two owned stimulus buffers.
    pub stimulus_frames: usize,
    /// Source-enforced payload budget, excluding allocator and OS overhead.
    pub budget_bytes: usize,
}

/// Hold and touch a large valid acquisition's production storage for measurement.
///
/// The visitor runs while every allocation remains alive. Storage uses the same
/// recording allocator and selected-channel ring constructor as native capture.
/// Two stimulus buffers represent reference and playback ownership. Native device
/// buffers and later analysis are excluded.
///
/// # Errors
/// Returns an error if the payload exceeds the acquisition budget or allocation fails.
pub fn visit_storage(visitor: impl FnOnce(StorageCase)) -> Result<(), String> {
    let inputs = 33;
    let rate = 48_000;
    let stimulus_frames = 3_000_000;
    // Native multi-device capture reserves ten seconds beyond the stimulus.
    let frames = stimulus_frames + rate as usize * 10;
    let budget_bytes = crate::limits::capture_bytes(inputs, frames, rate, stimulus_frames)?;
    let mut recordings = allocate_recordings(inputs, frames)?;
    for samples in &mut recordings {
        samples.resize(frames, 0.25);
    }
    let (mut producer, consumer) = input_ring(rate, inputs);
    while producer.push(0.25).is_ok() {}
    let reference = vec![0.25_f32; stimulus_frames];
    let playback = std::sync::Arc::new(reference.clone());
    visitor(StorageCase {
        inputs,
        frames,
        rate,
        stimulus_frames,
        budget_bytes,
    });
    black_box((&recordings, &producer, &consumer, &reference, &playback));
    Ok(())
}

/// Dimensions of the file-backed acquisition storage probe.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct FileStorageCase {
    /// Selected input count.
    pub inputs: usize,
    /// Frames per mono output WAV.
    pub frames: usize,
    /// Nominal sample rate.
    pub rate: u32,
    /// Frames in each of the two stimulus buffers.
    pub stimulus_frames: usize,
    /// Bounded live buffer budget, including bookkeeping allowance.
    pub budget_bytes: usize,
    /// Emitted sample payload bytes, excluding WAV headers.
    pub disk_payload_bytes: usize,
}

/// Measure live storage while streaming more than 512 MiB of raw audio to disk.
///
/// Uses the native spool, selected-channel ring, and stimulus ownership. The
/// accelerated producer waits between batches for the probe only. Callback
/// timing and disk throughput qualification are outside this memory measurement.
/// Temporary WAVs are finalized, length-checked, and removed before returning.
///
/// # Errors
/// Returns an error for allocation, disk, writer, or consistency failures.
pub fn visit_file_storage(visitor: impl FnOnce(FileStorageCase)) -> Result<(), String> {
    let inputs = 33;
    let rate = 384_000;
    let frames = crate::limits::MAX_CAPTURE_FRAMES;
    let stimulus_frames = frames - rate as usize * 10;
    let budget_bytes = crate::limits::file_capture_bytes(inputs, rate, stimulus_frames)?;
    let root = tempfile::tempdir().map_err(|error| error.to_string())?;
    // Worst case: every selected input owns an independent clock/diagnostic.
    let continuity: Vec<_> = (0..inputs)
        .map(|_| std::sync::Arc::new(super::multi_capture::continuity::Continuity::default()))
        .collect();
    let (mut producer, consumer) = input_ring(rate, inputs);
    while producer.push(0.25).is_ok() {}
    let reference = vec![0.25_f32; stimulus_frames];
    let playback = std::sync::Arc::new(reference.clone());
    super::multi_capture::probe_file_spool(root.path(), inputs, rate, frames, || {
        visitor(FileStorageCase {
            inputs,
            frames,
            rate,
            stimulus_frames,
            budget_bytes,
            disk_payload_bytes: frames * inputs * std::mem::size_of::<f32>(),
        });
    })?;
    black_box((&producer, &consumer, &reference, &playback, &continuity));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_covers_supported_formats_dimensions_and_input_rejections() {
        let mut seen = std::collections::BTreeSet::new();
        visit_callbacks(|case, callback, control| {
            assert!(
                seen.insert((case.core, case.format, case.channels, case.frames)),
                "duplicate case: {case:?}"
            );
            callback();
            control();
        });
        let formats = [
            (cpal::SampleFormat::F32, "f32"),
            (cpal::SampleFormat::F64, "f64"),
            (cpal::SampleFormat::I8, "i8"),
            (cpal::SampleFormat::I16, "i16"),
            (cpal::SampleFormat::I24, "i24"),
            (cpal::SampleFormat::I32, "i32"),
            (cpal::SampleFormat::I64, "i64"),
            (cpal::SampleFormat::U8, "u8"),
            (cpal::SampleFormat::U16, "u16"),
            (cpal::SampleFormat::U24, "u24"),
            (cpal::SampleFormat::U32, "u32"),
            (cpal::SampleFormat::U64, "u64"),
        ];
        for (native, format) in formats {
            assert!(super::super::measurement::measurement_sample_format_rank(native) < 12);
            for channels in [2, 8, 33, 94] {
                for frames in [32, 128, 512, 1024] {
                    for core in [
                        "sweep_input_with_meter",
                        "sweep_input_overflow",
                        "auxiliary_input_with_loopback",
                        "auxiliary_input_overflow",
                        "multidevice_input",
                        "multidevice_input_overflow",
                        "multidevice_output",
                        "auxiliary_output",
                        "sweep_input_malformed",
                        "auxiliary_input_malformed",
                        "multidevice_input_malformed",
                    ] {
                        assert!(
                            seen.contains(&(core, format, channels, frames)),
                            "missing {core}/{format}/{channels}/{frames}"
                        );
                    }
                    if matches!(native, cpal::SampleFormat::F32 | cpal::SampleFormat::F64) {
                        for core in [
                            "sweep_input_nonfinite",
                            "auxiliary_input_nonfinite",
                            "multidevice_input_nonfinite",
                        ] {
                            assert!(
                                seen.contains(&(core, format, channels, frames)),
                                "missing {core}/{format}/{channels}/{frames}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(seen.len(), 2208);
    }
}
