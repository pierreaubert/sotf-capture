//! Prepared native streams using the shared, fault-tested acquisition lifecycle.

// Rust guideline compliant 2026-02-21

use super::{RunningInput, drain_inputs};
use crate::signal_recorder::playback::SweepPlayback;
use crate::signal_recorder::presentation::PresentationClock;
use crate::signal_recorder::recorder::{NegotiatedInput, RecordedTake, SweepRecorder};
use std::cell::RefCell;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub(super) trait CaptureStream {
    fn play(&self) -> Result<(), String>;
    fn pause(&self) -> Result<(), String>;
}

impl CaptureStream for cpal::Stream {
    fn play(&self) -> Result<(), String> {
        cpal::traits::StreamTrait::play(self).map_err(|e| e.to_string())
    }
    fn pause(&self) -> Result<(), String> {
        cpal::traits::StreamTrait::pause(self).map_err(|e| e.to_string())
    }
}

struct CaptureInputs {
    inputs: Vec<RunningInput>,
    recordings: Vec<Vec<f32>>,
    limit: usize,
}

impl CaptureInputs {
    fn drain(&mut self) -> Result<(), String> {
        drain_inputs(&mut self.inputs, &mut self.recordings, self.limit)
    }
}

pub(super) struct PreparedRecorder {
    state: RefCell<CaptureInputs>,
    rate: u32,
    active: bool,
    shared_timeline: bool,
}

impl PreparedRecorder {
    pub fn new(
        inputs: Vec<RunningInput>,
        recordings: Vec<Vec<f32>>,
        rate: u32,
        limit: usize,
    ) -> Self {
        let shared_timeline = inputs.len() == 1;
        Self {
            state: RefCell::new(CaptureInputs {
                inputs,
                recordings,
                limit,
            }),
            rate,
            active: false,
            shared_timeline,
        }
    }
}

impl SweepRecorder for PreparedRecorder {
    fn start(
        &mut self,
        _: Option<&str>,
        _: &[u16],
        rate: u32,
        _: f64,
    ) -> Result<NegotiatedInput, String> {
        if self.active || rate != self.rate {
            return Err("invalid prepared input start".into());
        }
        let mut state = self.state.borrow_mut();
        let result = state.inputs.iter().try_for_each(|input| {
            input
                .stream
                .as_ref()
                .ok_or("input stream was released")?
                .play()
                .map_err(|error| format!("cannot start {}: {error}", input.name))
        });
        if let Err(error) = result {
            // Drop every prepared stream, including those started before the
            // failing one. A failed start requires no later stop from callers.
            for input in &mut state.inputs {
                input.stream.take();
            }
            return Err(error);
        }
        self.active = true;
        Ok(NegotiatedInput {
            hardware_channels: state.recordings.len(),
            sample_rate: self.rate,
        })
    }

    fn captured_samples(&self) -> usize {
        self.state.borrow().recordings.first().map_or(0, Vec::len)
    }

    fn input_frame_counts(&self, counts: &mut Vec<usize>) {
        counts.clear();
        counts.extend(self.state.borrow().recordings.iter().map(Vec::len));
    }

    fn shares_input_timeline(&self) -> bool {
        self.shared_timeline
    }

    fn check_health(&self) -> Result<(), String> {
        self.state.borrow_mut().drain()
    }

    fn stop(&mut self) -> Result<RecordedTake, String> {
        if !std::mem::replace(&mut self.active, false) {
            return Err("input stop without matching start".into());
        }
        let mut state = self.state.borrow_mut();
        let mut stop_error = None;
        for input in &mut state.inputs {
            if let Some(stream) = input.stream.take() {
                if let Err(error) = stream.pause() {
                    stop_error
                        .get_or_insert_with(|| format!("cannot stop {}: {error}", input.name));
                }
                drop(stream);
            }
        }
        let drained = state.drain();
        if let Some(error) = stop_error {
            return Err(error);
        }
        drained?;
        Ok(RecordedTake {
            recorded_per_channel: std::mem::take(&mut state.recordings),
            analysis_sample_rate: self.rate,
            dropped_samples: 0,
        })
    }
}

pub(super) struct PreparedPlayback {
    stream: Option<Box<dyn CaptureStream>>,
    failed: Arc<AtomicBool>,
    presentation: Arc<PresentationClock>,
}

impl PreparedPlayback {
    pub fn new(
        stream: Box<dyn CaptureStream>,
        failed: Arc<AtomicBool>,
        presentation: Arc<PresentationClock>,
    ) -> Self {
        Self {
            stream: Some(stream),
            failed,
            presentation,
        }
    }
}

impl SweepPlayback for PreparedPlayback {
    fn start(&mut self, _: &Path, _: Option<&str>, _: u16, _: u32) -> Result<(), String> {
        if let Err(error) = self
            .stream
            .as_ref()
            .ok_or("output stream was released")?
            .play()
        {
            self.stream.take();
            return Err(format!("cannot start playback: {error}"));
        }
        Ok(())
    }
    fn is_finished(&mut self) -> bool {
        self.presentation.is_finished()
    }
    fn check_health(&self) -> Result<(), String> {
        if self.failed.load(Ordering::Acquire) {
            Err("capture playback stream failed".into())
        } else {
            Ok(())
        }
    }
    fn stop(&mut self) -> Result<(), String> {
        let result = self.stream.take().map_or(Ok(()), |stream| {
            let result = stream.pause();
            drop(stream);
            result
        });
        result?;
        self.check_health()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    struct Stream {
        id: usize,
        trace: Rc<RefCell<Vec<String>>>,
        start_fails: bool,
        stop_fails: bool,
    }
    impl CaptureStream for Stream {
        fn play(&self) -> Result<(), String> {
            self.trace.borrow_mut().push(format!("play{}", self.id));
            if self.start_fails {
                Err("injected start failure".into())
            } else {
                Ok(())
            }
        }
        fn pause(&self) -> Result<(), String> {
            self.trace.borrow_mut().push(format!("pause{}", self.id));
            if self.stop_fails {
                Err("injected stop failure".into())
            } else {
                Ok(())
            }
        }
    }
    impl Drop for Stream {
        fn drop(&mut self) {
            self.trace.borrow_mut().push(format!("drop{}", self.id));
        }
    }
    fn input(
        id: usize,
        trace: &Rc<RefCell<Vec<String>>>,
        start_fails: bool,
        stop_fails: bool,
    ) -> RunningInput {
        let (_, consumer) = rtrb::RingBuffer::new(16);
        RunningInput {
            stream: Some(Box::new(Stream {
                id,
                trace: Rc::clone(trace),
                start_fails,
                stop_fails,
            })),
            consumer,
            routes: vec![(id, 0)],
            failed: Arc::new(AtomicBool::new(false)),
            name: format!("input{id}"),
        }
    }

    #[test]
    fn partial_start_failure_releases_started_failed_and_unstarted_streams_once() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let inputs = (0..3).map(|id| input(id, &trace, id == 1, false)).collect();
        let mut recorder = PreparedRecorder::new(inputs, vec![Vec::new(); 3], 48000, 16);
        assert!(recorder.start(None, &[0, 1, 2], 48000, 0.1).is_err());
        assert_eq!(
            *trace.borrow(),
            ["play0", "play1", "drop0", "drop1", "drop2"]
        );
        drop(recorder);
        assert_eq!(trace.borrow().len(), 5);
    }

    #[test]
    fn stop_failure_still_stops_and_releases_every_input() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let inputs = (0..3).map(|id| input(id, &trace, false, id == 0)).collect();
        let mut recorder = PreparedRecorder::new(inputs, vec![Vec::new(); 3], 48000, 16);
        recorder.start(None, &[0, 1, 2], 48000, 0.1).unwrap();
        trace.borrow_mut().clear();
        assert!(recorder.stop().is_err());
        assert_eq!(
            *trace.borrow(),
            ["pause0", "drop0", "pause1", "drop1", "pause2", "drop2"]
        );
        assert!(recorder.stop().is_err());
        drop(recorder);
        assert_eq!(trace.borrow().len(), 6);
    }

    #[test]
    fn independent_ring_lengths_and_origins_survive_the_prepared_backend() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let mut inputs = Vec::new();
        let mut producers = Vec::new();
        for id in 0..2 {
            let (mut producer, consumer) = rtrb::RingBuffer::new(16);
            for frame in 0..id + 3 {
                producer
                    .push(f32::from(u16::try_from(frame).unwrap()))
                    .unwrap();
            }
            let mut prepared = input(id, &trace, false, false);
            prepared.consumer = consumer;
            inputs.push(prepared);
            producers.push(producer);
        }
        let mut recorder = PreparedRecorder::new(inputs, vec![Vec::new(); 2], 48000, 16);
        recorder.start(None, &[0, 1], 48000, 0.1).unwrap();
        recorder.check_health().unwrap();
        let mut counts = Vec::new();
        recorder.input_frame_counts(&mut counts);
        assert_eq!(counts, [3, 4]);
        assert!(!recorder.shares_input_timeline());
        producers[0].push(99.0).unwrap();
        let take = recorder.stop().unwrap();
        assert_eq!(
            take.recorded_per_channel,
            [vec![0.0, 1.0, 2.0, 99.0], vec![0.0, 1.0, 2.0, 3.0]]
        );
    }

    #[test]
    fn failed_output_start_releases_its_prepared_stream() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let stream = Stream {
            id: 0,
            trace: Rc::clone(&trace),
            start_fails: true,
            stop_fails: false,
        };
        let mut output = PreparedPlayback::new(
            Box::new(stream),
            Arc::new(AtomicBool::new(false)),
            Arc::new(PresentationClock::new()),
        );
        assert!(output.start(Path::new("prepared"), None, 0, 48000).is_err());
        output.stop().unwrap();
        drop(output);
        assert_eq!(*trace.borrow(), ["play0", "drop0"]);
    }
}
