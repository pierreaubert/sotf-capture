//! Heap/logger/deadline probes for rejected callback input blocks.

// Rust guideline compliant 2026-02-21

use super::{
    CallbackCase, InputCounters, InputRings, capture_channels, capture_frames_tracked,
    capture_pairs,
};
use std::hint::black_box;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Clone, Copy)]
pub(super) enum Fault {
    Malformed,
    Nonfinite,
}

impl Fault {
    fn names(self) -> [&'static str; 3] {
        match self {
            Self::Malformed => [
                "sweep_input_malformed",
                "auxiliary_input_malformed",
                "multidevice_input_malformed",
            ],
            Self::Nonfinite => [
                "sweep_input_nonfinite",
                "auxiliary_input_nonfinite",
                "multidevice_input_nonfinite",
            ],
        }
    }
}

pub(super) fn visit_nonfinite<T>(
    format: &'static str,
    channels: usize,
    frames: usize,
    visitor: &mut impl FnMut(CallbackCase, &mut dyn FnMut(), &mut dyn FnMut()),
) where
    T: cpal::SizedSample + cpal::FromSample<f32>,
    f32: cpal::FromSample<T>,
{
    let selected = channels.min(33);
    let mut input = vec![T::from_sample_(0.25); channels * frames];
    // Scan a complete valid prefix before finding NaN in the final frame.
    input[(frames - 1) * channels + selected - 1] = T::from_sample_(f32::NAN);
    input[frames * channels - 1] = T::from_sample_(f32::NAN);
    visit(
        &input,
        CallbackCase {
            core: "",
            format,
            channels,
            selected,
            frames,
        },
        Fault::Nonfinite,
        visitor,
    );
}

pub(super) fn visit<T>(
    input: &[T],
    case: CallbackCase,
    fault: Fault,
    visitor: &mut impl FnMut(CallbackCase, &mut dyn FnMut(), &mut dyn FnMut()),
) where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let [sweep, auxiliary, multi] = fault.names();
    visit_sweep(
        input,
        CallbackCase {
            core: sweep,
            ..case
        },
        visitor,
    );
    visit_auxiliary(
        input,
        CallbackCase {
            core: auxiliary,
            selected: 2,
            ..case
        },
        visitor,
    );
    visit_multi(
        input,
        CallbackCase {
            core: multi,
            ..case
        },
        visitor,
    );
}

fn visit_sweep<T>(
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
    let counts: Vec<_> = (0..case.selected)
        .map(|_| Arc::new(AtomicUsize::new(0)))
        .collect();
    let channels: Vec<_> = (0..case.selected).collect();
    let clocks: Vec<_> = (0..case.selected)
        .map(|_| Arc::new(super::super::continuity::Continuity::default()))
        .collect();
    let failed = AtomicBool::new(false);
    let lost = AtomicUsize::new(0);
    let meter = super::super::level_meter::CaptureLevelMeter::new();
    visitor(
        case,
        &mut || {
            capture_channels(
                black_box(input),
                case.channels,
                &mut InputRings {
                    producers: &mut producers,
                    counts: &counts,
                    channels: &channels,
                    overruns: &lost,
                    failed: &failed,
                    meter: Some(&meter),
                    continuity: &clocks,
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
    assert!(failed.load(Ordering::Relaxed), "{case:?}");
    assert!(
        counts
            .iter()
            .all(|count| count.load(Ordering::Relaxed) == 0),
        "rejected sweep block published samples: {case:?}"
    );
    assert_eq!(lost.load(Ordering::Relaxed), 0, "{case:?}");
}

fn visit_auxiliary<T>(
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
    let clock = super::super::continuity::Continuity::default();
    visitor(
        case,
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
                    continuity: &clock,
                },
            );
        },
        &mut || {
            while let Ok(pair) = consumer.pop() {
                black_box(pair);
            }
        },
    );
    assert!(failed.load(Ordering::Relaxed), "{case:?}");
    assert_eq!(accepted.load(Ordering::Relaxed), 0, "{case:?}");
    assert_eq!(lost.load(Ordering::Relaxed), 0, "{case:?}");
}

fn visit_multi<T>(
    input: &[T],
    case: CallbackCase,
    visitor: &mut impl FnMut(CallbackCase, &mut dyn FnMut(), &mut dyn FnMut()),
) where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let selected: Vec<_> = (0..case.selected)
        .map(|channel| u16::try_from(channel).expect("bounded hardware width"))
        .collect();
    let (mut producer, mut consumer) = rtrb::RingBuffer::new(case.frames * case.selected);
    let failed = AtomicBool::new(false);
    let clock = super::super::continuity::Continuity::default();
    visitor(
        case,
        &mut || {
            capture_frames_tracked(
                black_box(input),
                case.channels,
                &selected,
                &mut producer,
                &failed,
                &clock,
            );
        },
        &mut || {
            while let Ok(sample) = consumer.pop() {
                black_box(sample);
            }
        },
    );
    assert!(failed.load(Ordering::Relaxed), "{case:?}");
    assert_eq!(
        clock.snapshot().missing_frames_per_channel,
        0,
        "rejected block was replaced by an overflow case: {case:?}"
    );
}
