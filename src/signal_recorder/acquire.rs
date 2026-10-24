//! Shared measurement stream lifecycle with explicit readiness and completion gates.

// Rust guideline compliant 2026-02-21

use super::playback::SweepPlayback;
use super::recorder::{RecordedTake, SweepRecorder};
use super::types::{CancelFlag, cancel_requested};
use std::path::Path;
use std::time::{Duration, Instant};

// Every control wait remains responsive to cancellation and callback errors.
const POLL: Duration = Duration::from_millis(10);
const STARTUP: Duration = Duration::from_secs(5);
const TAIL: Duration = Duration::from_secs(1);

pub(super) trait Clock {
    fn now(&self) -> Duration;
    fn sleep(&mut self, duration: Duration);
}

struct SystemClock(Instant);
impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
    fn sleep(&mut self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

/// Private lifecycle boundary shared by memory and file-backed inputs.
pub(super) trait CaptureInput {
    type Output;
    fn start(
        &mut self,
        device: Option<&str>,
        channels: &[u16],
        rate: u32,
        duration: f64,
    ) -> Result<super::recorder::NegotiatedInput, String>;
    fn input_frame_counts(&self, counts: &mut Vec<usize>);
    fn check_health(&self) -> Result<(), String>;
    fn stop(&mut self) -> Result<Self::Output, String>;
    fn validate(&self, take: &Self::Output, rate: u32, channels: usize) -> Result<(), String>;
}

impl<R: SweepRecorder> CaptureInput for R {
    type Output = RecordedTake;
    fn start(
        &mut self,
        device: Option<&str>,
        channels: &[u16],
        rate: u32,
        duration: f64,
    ) -> Result<super::recorder::NegotiatedInput, String> {
        SweepRecorder::start(self, device, channels, rate, duration)
    }
    fn input_frame_counts(&self, counts: &mut Vec<usize>) {
        SweepRecorder::input_frame_counts(self, counts);
    }
    fn check_health(&self) -> Result<(), String> {
        SweepRecorder::check_health(self)
    }
    fn stop(&mut self) -> Result<Self::Output, String> {
        SweepRecorder::stop(self)
    }
    fn validate(&self, take: &RecordedTake, rate: u32, channels: usize) -> Result<(), String> {
        validate_recorded_take(take, rate, channels, self.shares_input_timeline())
    }
}

pub(super) fn validate_recorded_take(
    take: &RecordedTake,
    rate: u32,
    channels: usize,
    shared_timeline: bool,
) -> Result<(), String> {
    if take.dropped_samples != 0 {
        return Err(format!(
            "capture lost {} input samples",
            take.dropped_samples
        ));
    }
    if take.analysis_sample_rate != rate
        || take.recorded_per_channel.len() != channels
        || take.recorded_per_channel.iter().any(|samples| {
            samples.is_empty()
                || samples.len() > crate::limits::MAX_CAPTURE_FRAMES
                || (shared_timeline && samples.len() != take.recorded_per_channel[0].len())
                || samples.iter().any(|sample| !sample.is_finite())
        })
    {
        return Err("capture backend returned inconsistent or invalid input samples".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CaptureRequest<'a> {
    pub wav: &'a Path,
    pub frames: usize,
    pub rate: u32,
    pub output_channel: u16,
    pub input_channels: &'a [u16],
    pub output_device: Option<&'a str>,
    pub input_device: Option<&'a str>,
}

pub(super) fn capture<R: CaptureInput>(
    playback: &mut impl SweepPlayback,
    recorder: &mut R,
    request: CaptureRequest<'_>,
    cancel: Option<&CancelFlag>,
) -> Result<R::Output, String> {
    capture_with_clock(
        playback,
        recorder,
        request,
        cancel,
        &mut SystemClock(Instant::now()),
    )
}

fn check_cancel(cancel: Option<&CancelFlag>) -> Result<(), String> {
    if cancel_requested(cancel) {
        Err(super::consts::CANCELLED_ERR.into())
    } else {
        Ok(())
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "The request frame count is checked against the 16-million-frame limit."
)]
pub(super) fn capture_with_clock<R: CaptureInput>(
    playback: &mut impl SweepPlayback,
    recorder: &mut R,
    request: CaptureRequest<'_>,
    cancel: Option<&CancelFlag>,
    clock: &mut impl Clock,
) -> Result<R::Output, String> {
    check_cancel(cancel)?;
    if request.rate == 0
        || request.frames == 0
        || request.frames > crate::limits::MAX_CAPTURE_FRAMES
        || request.input_channels.is_empty()
    {
        return Err("invalid capture request".into());
    }
    let duration = Duration::from_secs_f64(request.frames as f64 / f64::from(request.rate));
    let negotiated = recorder.start(
        request.input_device,
        request.input_channels,
        request.rate,
        duration.as_secs_f64(),
    )?;
    let mut playback_started = false;
    let mut counts = Vec::with_capacity(request.input_channels.len());
    let result = (|| -> Result<(), String> {
        if negotiated.sample_rate == 0 {
            return Err("invalid negotiated input rate".into());
        }
        wait_for_input(
            recorder,
            negotiated.sample_rate,
            request.input_channels.len(),
            cancel,
            clock,
            &mut counts,
        )?;
        check_cancel(cancel)?;
        playback.start(
            request.wav,
            request.output_device,
            request.output_channel,
            request.rate,
        )?;
        playback_started = true;
        let deadline = clock.now() + duration + Duration::from_secs(3);
        loop {
            check_cancel(cancel)?;
            recorder.check_health()?;
            playback.check_health()?;
            if playback.is_finished() {
                break;
            }
            if clock.now() >= deadline {
                return Err("playback timed out before completion".into());
            }
            clock.sleep(POLL);
        }
        capture_tail(
            playback,
            recorder,
            negotiated.sample_rate,
            cancel,
            clock,
            &mut counts,
        )?;
        Ok(())
    })();
    // Once input starts, every path stops it exactly once. Always attempt both
    // stops, including when one fails; cancellation takes precedence afterward.
    let playback_stop = if playback_started {
        playback.stop()
    } else {
        Ok(())
    };
    let take = recorder.stop();
    check_cancel(cancel)?;
    result?;
    super::record::cancel_aware_stop(playback_stop, false)?;
    let take = take?;
    recorder.validate(&take, negotiated.sample_rate, request.input_channels.len())?;
    Ok(take)
}

fn wait_for_input(
    recorder: &impl CaptureInput,
    sample_rate: u32,
    max_clocks: usize,
    cancel: Option<&CancelFlag>,
    clock: &mut impl Clock,
    counts: &mut Vec<usize>,
) -> Result<(), String> {
    let deadline = clock.now() + STARTUP;
    let mut input_clocks = None;
    loop {
        check_cancel(cancel)?;
        recorder.check_health()?;
        recorder.input_frame_counts(counts);
        if counts.is_empty()
            || counts.len() > max_clocks
            || input_clocks.is_some_and(|expected| counts.len() != expected)
        {
            return Err("capture backend returned inconsistent input clock identities".into());
        }
        // Never let a missing clock make the remaining clocks appear ready.
        input_clocks = Some(counts.len());
        if counts
            .iter()
            .all(|frames| *frames >= (sample_rate as usize / 4).max(1))
        {
            return Ok(());
        }
        if clock.now() >= deadline {
            return Err("input readiness timed out".into());
        }
        clock.sleep(POLL);
    }
}

fn capture_tail(
    playback: &impl SweepPlayback,
    recorder: &impl CaptureInput,
    sample_rate: u32,
    cancel: Option<&CancelFlag>,
    clock: &mut impl Clock,
    counts: &mut Vec<usize>,
) -> Result<(), String> {
    let tail_started = clock.now();
    let clocks = counts.len();
    recorder.input_frame_counts(counts);
    if counts.len() != clocks || clocks == 0 {
        return Err("input clock identities changed before drain".into());
    }
    let targets = counts
        .iter()
        .map(|frames| {
            frames
                .checked_add(sample_rate as usize)
                .ok_or("input drain frame count overflow")
        })
        .collect::<Result<Vec<_>, _>>()?;
    loop {
        check_cancel(cancel)?;
        recorder.check_health()?;
        playback.check_health()?;
        recorder.input_frame_counts(counts);
        if counts.len() != targets.len() || counts.is_empty() {
            return Err("input clock identities changed during capture".into());
        }
        if clock.now() >= tail_started + TAIL
            && counts
                .iter()
                .zip(&targets)
                .all(|(count, target)| count >= target)
        {
            break;
        }
        if clock.now() >= tail_started + TAIL + STARTUP {
            return Err("input drain timed out after output presentation".into());
        }
        clock.sleep(POLL);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_recorder::recorder::NegotiatedInput;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CancelBoundary {
        OutputStart,
        InputStop,
    }

    #[derive(Default)]
    struct Schedule {
        now: Cell<Duration>,
        input_started: Cell<Option<Duration>>,
        output_started: Cell<Option<Duration>>,
        first_input_after: Duration,
        independent_input_delays: Vec<Duration>,
        completion_after: Option<Duration>,
        input_failure_at: Option<Duration>,
        input_stalls_at: Option<Duration>,
        input_disappears_at: Option<Duration>,
        output_failure_at: Option<Duration>,
        cancel_at: Option<Duration>,
        cancel: CancelFlag,
        events: RefCell<Vec<(&'static str, Duration)>>,
        output_start_fails: bool,
        output_stop_fails: bool,
        input_stop_fails: bool,
        cancel_on: Option<CancelBoundary>,
        polling_seed: Cell<u64>,
        callback_frames: usize,
        lost_samples: u64,
    }

    impl Schedule {
        fn event(&self, event: &'static str) {
            self.events.borrow_mut().push((event, self.now.get()));
        }
    }

    struct VirtualClock(Rc<Schedule>);
    impl Clock for VirtualClock {
        fn now(&self) -> Duration {
            self.0.now.get()
        }
        fn sleep(&mut self, duration: Duration) {
            let duration = if self.0.polling_seed.get() == 0 {
                duration
            } else {
                Duration::from_millis(1 + random(&self.0.polling_seed) % 31)
            };
            self.0.now.set(self.0.now.get() + duration);
            if self
                .0
                .cancel_at
                .is_some_and(|time| self.0.now.get() >= time)
            {
                self.0.cancel.store(true, Ordering::Relaxed);
            }
        }
    }

    struct Input(Rc<Schedule>);
    impl Input {
        fn count(&self, index: usize) -> usize {
            let started = self.0.input_started.get().unwrap();
            let observed = if index == 0 {
                self.0
                    .input_stalls_at
                    .map_or(self.0.now.get(), |time| time.min(self.0.now.get()))
            } else {
                self.0.now.get()
            };
            let delay = self
                .0
                .independent_input_delays
                .get(index)
                .copied()
                .unwrap_or(self.0.first_input_after);
            let elapsed = observed.saturating_sub(started + delay);
            let frames = usize::try_from(elapsed.as_nanos() * 48_000 / 1_000_000_000).unwrap();
            let block = self.0.callback_frames.max(1);
            frames / block * block
        }
    }
    impl SweepRecorder for Input {
        fn start(
            &mut self,
            _: Option<&str>,
            _: &[u16],
            _: u32,
            _: f64,
        ) -> Result<NegotiatedInput, String> {
            self.0.event("input.start");
            self.0.input_started.set(Some(self.0.now.get()));
            Ok(NegotiatedInput {
                hardware_channels: 2,
                sample_rate: 48000,
            })
        }
        fn captured_samples(&self) -> usize {
            self.count(0)
        }
        fn input_frame_counts(&self, counts: &mut Vec<usize>) {
            counts.clear();
            counts.extend(
                (0..self.0.independent_input_delays.len().max(1)).map(|index| self.count(index)),
            );
            if self
                .0
                .input_disappears_at
                .is_some_and(|time| self.0.now.get() >= time)
            {
                counts.pop();
            }
        }
        fn shares_input_timeline(&self) -> bool {
            self.0.independent_input_delays.is_empty()
        }
        fn check_health(&self) -> Result<(), String> {
            if self
                .0
                .input_failure_at
                .is_some_and(|time| self.0.now.get() >= time)
            {
                Err("injected input failure".into())
            } else {
                Ok(())
            }
        }
        fn stop(&mut self) -> Result<RecordedTake, String> {
            self.0.event("input.stop");
            let samples = (0..self.0.independent_input_delays.len().max(1))
                .map(|index| vec![0.1; self.count(index)])
                .collect();
            assert!(
                self.0.input_started.take().is_some(),
                "input must stop exactly once"
            );
            if self.0.input_stop_fails {
                return Err("injected input stop failure".into());
            }
            if self.0.cancel_on == Some(CancelBoundary::InputStop) {
                self.0.cancel.store(true, Ordering::Relaxed);
            }
            SweepRecorder::check_health(self)?;
            Ok(RecordedTake {
                recorded_per_channel: samples,
                analysis_sample_rate: 48000,
                dropped_samples: self.0.lost_samples,
            })
        }
    }

    struct Output(Rc<Schedule>);
    impl SweepPlayback for Output {
        fn start(&mut self, _: &Path, _: Option<&str>, _: u16, _: u32) -> Result<(), String> {
            self.0.event("output.start");
            if self.0.output_start_fails {
                return Err("injected output start failure".into());
            }
            self.0.output_started.set(Some(self.0.now.get()));
            if self.0.cancel_on == Some(CancelBoundary::OutputStart) {
                self.0.cancel.store(true, Ordering::Relaxed);
            }
            Ok(())
        }
        fn is_finished(&mut self) -> bool {
            self.0.completion_after.is_some_and(|duration| {
                self.0.now.get() >= self.0.output_started.get().unwrap() + duration
            })
        }
        fn check_health(&self) -> Result<(), String> {
            if self
                .0
                .output_failure_at
                .is_some_and(|time| self.0.now.get() >= time)
            {
                Err("injected output failure".into())
            } else {
                Ok(())
            }
        }
        fn stop(&mut self) -> Result<(), String> {
            self.0.event("output.stop");
            assert!(
                self.0.output_started.take().is_some(),
                "output must stop exactly once"
            );
            if self.0.output_stop_fails {
                Err("injected output stop failure".into())
            } else {
                self.check_health()
            }
        }
    }

    fn run(schedule: Schedule) -> (Result<RecordedTake, String>, Rc<Schedule>) {
        let schedule = Rc::new(schedule);
        let channels: Vec<_> = (0..schedule.independent_input_delays.len().max(1))
            .map(|index| u16::try_from(index).unwrap())
            .collect();
        let result = capture_with_clock(
            &mut Output(Rc::clone(&schedule)),
            &mut Input(Rc::clone(&schedule)),
            CaptureRequest {
                wav: Path::new("virtual.wav"),
                frames: 4800,
                rate: 48000,
                output_channel: 0,
                input_channels: &channels,
                input_device: None,
                output_device: None,
            },
            Some(&schedule.cancel),
            &mut VirtualClock(Rc::clone(&schedule)),
        );
        (result, schedule)
    }

    fn healthy() -> Schedule {
        Schedule {
            completion_after: Some(Duration::from_millis(100)),
            cancel: Arc::new(AtomicBool::new(false)),
            ..Schedule::default()
        }
    }

    // A fixed xorshift generator keeps failures reproducible without a new
    // dependency. Different seeds vary callback blocks and control wakeups.
    fn random(state: &Cell<u64>) -> u64 {
        let mut value = state.get();
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        state.set(value);
        value
    }

    fn seeded(seed: u64) -> Schedule {
        let state = Cell::new(seed);
        Schedule {
            independent_input_delays: vec![
                Duration::from_millis(random(&state) % 400),
                Duration::from_millis(random(&state) % 900),
            ],
            callback_frames: usize::try_from(1 + random(&state) % 1024).unwrap(),
            // The requested stimulus is 100 ms; add a variable presentation delay.
            completion_after: Some(Duration::from_millis(100 + random(&state) % 400)),
            polling_seed: state,
            ..healthy()
        }
    }

    fn assert_cleanup(schedule: &Schedule, seed: u64) {
        let events = schedule.events.borrow();
        for (start, stop) in [
            ("input.start", "input.stop"),
            ("output.start", "output.stop"),
        ] {
            let starts = events.iter().filter(|(event, _)| *event == start).count();
            let stops = events.iter().filter(|(event, _)| *event == stop).count();
            let failed_start = usize::from(start == "output.start" && schedule.output_start_fails);
            assert_eq!(starts - failed_start, stops, "seed {seed}: {events:?}");
            assert!(starts <= 1, "seed {seed}: {events:?}");
        }
        assert!(schedule.input_started.get().is_none(), "seed {seed}");
        assert!(schedule.output_started.get().is_none(), "seed {seed}");
    }

    #[test]
    fn seeded_schedules_preserve_readiness_presentation_and_failure_outcomes() {
        for seed in 1..=256 {
            let (result, baseline) = run(seeded(seed));
            let take = result.unwrap_or_else(|error| panic!("healthy seed {seed}: {error}"));
            let events = baseline.events.borrow();
            let output_start = events
                .iter()
                .find(|(name, _)| *name == "output.start")
                .unwrap()
                .1;
            let stopped = events
                .iter()
                .find(|(name, _)| *name == "input.stop")
                .unwrap()
                .1;
            let presented = output_start + baseline.completion_after.unwrap();
            for (index, delay) in baseline.independent_input_delays.iter().enumerate() {
                assert!(
                    output_start >= *delay + Duration::from_millis(250),
                    "seed {seed}"
                );
                assert!(stopped >= presented + Duration::from_secs(1), "seed {seed}");
                let frames_at_presentation = usize::try_from(
                    presented.saturating_sub(*delay).as_nanos() * 48_000 / 1_000_000_000,
                )
                .unwrap();
                // One incomplete callback block may still be pending at the
                // physical presentation boundary, but a full tail must follow.
                assert!(
                    take.recorded_per_channel[index].len() + baseline.callback_frames
                        >= frames_at_presentation + 48_000,
                    "seed {seed}"
                );
            }
            assert_cleanup(&baseline, seed);

            for fault in 0..12 {
                let mut schedule = seeded(seed);
                let during_playback = output_start + baseline.completion_after.unwrap() / 2;
                let during_tail = presented + Duration::from_millis(300);
                match fault {
                    0 => schedule.cancel_at = Some(output_start / 2),
                    1 => schedule.cancel_at = Some(during_playback),
                    2 => schedule.cancel_at = Some(during_tail),
                    3 => schedule.input_failure_at = Some(output_start / 2),
                    4 => schedule.input_failure_at = Some(during_tail),
                    5 => schedule.output_failure_at = Some(during_playback),
                    6 => schedule.output_failure_at = Some(during_tail),
                    7 => schedule.completion_after = None,
                    8 => schedule.input_stalls_at = Some(during_tail),
                    9 => schedule.lost_samples = 1,
                    10 => schedule.input_stop_fails = true,
                    11 => schedule.output_stop_fails = true,
                    _ => unreachable!(),
                }
                let (result, schedule) = run(schedule);
                let error = result.expect_err(&format!("seed {seed}, fault {fault} accepted"));
                if fault <= 2 {
                    assert_eq!(error, super::super::consts::CANCELLED_ERR, "seed {seed}");
                }
                assert_cleanup(&schedule, seed);
            }
        }
    }

    #[test]
    fn cancellation_at_start_and_cleanup_boundaries_never_accepts_a_take() {
        for boundary in [CancelBoundary::OutputStart, CancelBoundary::InputStop] {
            let (result, schedule) = run(Schedule {
                cancel_on: Some(boundary),
                ..healthy()
            });
            assert_eq!(result.unwrap_err(), super::super::consts::CANCELLED_ERR);
            assert_cleanup(&schedule, 0);
        }
    }

    #[test]
    fn disappearing_input_during_readiness_cannot_start_output() {
        let (result, schedule) = run(Schedule {
            independent_input_delays: vec![Duration::ZERO, Duration::from_millis(700)],
            input_disappears_at: Some(Duration::from_millis(100)),
            ..healthy()
        });
        assert!(
            result.is_err(),
            "a missing input clock must invalidate capture"
        );
        assert!(
            !schedule
                .events
                .borrow()
                .iter()
                .any(|(event, _)| *event == "output.start")
        );
        assert_cleanup(&schedule, 0);
    }

    #[test]
    fn output_waits_for_real_input_and_both_streams_stop_once() {
        let (result, schedule) = run(Schedule {
            first_input_after: Duration::from_millis(700),
            ..healthy()
        });
        result.unwrap();
        assert_eq!(
            *schedule.events.borrow(),
            vec![
                ("input.start", Duration::ZERO),
                ("output.start", Duration::from_millis(950)),
                ("output.stop", Duration::from_millis(2050)),
                ("input.stop", Duration::from_millis(2050)),
            ]
        );
    }

    #[test]
    fn absent_input_never_starts_output() {
        let (result, schedule) = run(Schedule {
            first_input_after: Duration::from_secs(30),
            ..healthy()
        });
        assert!(result.unwrap_err().contains("readiness timed out"));
        assert_eq!(
            *schedule.events.borrow(),
            vec![
                ("input.start", Duration::ZERO),
                ("input.stop", Duration::from_secs(5))
            ]
        );
    }

    #[test]
    fn independent_inputs_keep_their_origins_and_each_must_deliver_a_tail() {
        let (result, _) = run(Schedule {
            independent_input_delays: vec![Duration::ZERO, Duration::from_millis(700)],
            ..healthy()
        });
        let take = result.unwrap();
        assert_eq!(
            take.recorded_per_channel[0].len() - take.recorded_per_channel[1].len(),
            33_600
        );
        // The earlier stream already has more frames at presentation. A
        // minimum aggregate counter could hide its subsequent stall.
        let (result, _) = run(Schedule {
            independent_input_delays: vec![Duration::ZERO, Duration::from_millis(700)],
            input_stalls_at: Some(Duration::from_millis(1100)),
            ..healthy()
        });
        assert!(result.unwrap_err().contains("input drain timed out"));
    }

    #[test]
    fn timeout_and_asynchronous_errors_cannot_become_success() {
        for schedule in [
            Schedule {
                completion_after: None,
                ..healthy()
            },
            Schedule {
                input_failure_at: Some(Duration::from_millis(400)),
                ..healthy()
            },
            Schedule {
                output_failure_at: Some(Duration::from_millis(400)),
                ..healthy()
            },
            Schedule {
                lost_samples: 1234,
                ..healthy()
            },
            Schedule {
                input_stalls_at: Some(Duration::from_millis(300)),
                ..healthy()
            },
        ] {
            let (result, schedule) = run(schedule);
            assert!(result.is_err());
            let events = schedule.events.borrow();
            assert_eq!(
                events
                    .iter()
                    .filter(|(event, _)| *event == "input.stop")
                    .count(),
                1
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|(event, _)| *event == "output.stop")
                    .count(),
                1
            );
        }
    }

    #[test]
    fn cancellation_survives_startup_playback_tail_and_stop_failure() {
        for millis in [100, 300, 500, 1340] {
            let (result, schedule) = run(Schedule {
                cancel_at: Some(Duration::from_millis(millis)),
                input_stop_fails: true,
                output_stop_fails: true,
                ..healthy()
            });
            assert_eq!(result.unwrap_err(), super::super::consts::CANCELLED_ERR);
            assert_eq!(
                schedule
                    .events
                    .borrow()
                    .iter()
                    .filter(|(event, _)| *event == "input.stop")
                    .count(),
                1
            );
        }
    }

    #[test]
    fn failed_output_start_still_stops_input() {
        let (result, schedule) = run(Schedule {
            output_start_fails: true,
            ..healthy()
        });
        assert!(result.unwrap_err().contains("output start"));
        assert_eq!(
            schedule
                .events
                .borrow()
                .iter()
                .map(|(event, _)| *event)
                .collect::<Vec<_>>(),
            ["input.start", "output.start", "input.stop"]
        );
    }
}
