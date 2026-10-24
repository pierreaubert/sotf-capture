//! Prepared native streams using the shared, fault-tested acquisition lifecycle.

// Rust guideline compliant 2026-02-21

use super::spool::{CapturedFile, FileSpool};
use super::{RunningInput, drain_storage};
use crate::signal_recorder::acquire::{CaptureInput, validate_recorded_take};
use crate::signal_recorder::playback::SweepPlayback;
use crate::signal_recorder::presentation::PresentationClock;
use crate::signal_recorder::recorder::{NegotiatedInput, RecordedTake};
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

pub(super) enum Storage {
    Memory(Vec<Vec<f32>>),
    Files(FileSpool),
}

#[derive(Debug)]
pub(super) enum CapturedStorage {
    Memory(RecordedTake),
    Files(Vec<CapturedFile>),
}

impl Storage {
    pub fn channels(&self) -> usize {
        match self {
            Self::Memory(recordings) => recordings.len(),
            Self::Files(spool) => spool.counts.len(),
        }
    }
    pub fn count(&self, channel: usize) -> usize {
        match self {
            Self::Memory(recordings) => recordings[channel].len(),
            Self::Files(spool) => spool.counts[channel],
        }
    }
    pub fn push(&mut self, channel: usize, sample: f32) -> Result<(), String> {
        if !sample.is_finite() {
            return Err("capture ring contains non-finite samples".into());
        }
        match self {
            Self::Memory(recordings) => {
                recordings[channel].push(sample);
                Ok(())
            }
            Self::Files(spool) => spool.push(channel, sample),
        }
    }
    fn check_health(&self) -> Result<(), String> {
        match self {
            Self::Memory(_) => Ok(()),
            Self::Files(spool) => spool.check_health(),
        }
    }
    fn finish(&mut self, rate: u32) -> Result<CapturedStorage, String> {
        match self {
            Self::Memory(recordings) => Ok(CapturedStorage::Memory(RecordedTake {
                recorded_per_channel: std::mem::take(recordings),
                analysis_sample_rate: rate,
                dropped_samples: 0,
            })),
            Self::Files(spool) => spool.finish().map(CapturedStorage::Files),
        }
    }
}

struct CaptureInputs {
    inputs: Vec<RunningInput>,
    storage: Storage,
    limit: usize,
}

impl CaptureInputs {
    fn drain(&mut self) -> Result<(), String> {
        self.storage.check_health()?;
        drain_storage(&mut self.inputs, &mut self.storage, self.limit)
    }
}

pub(super) struct PreparedRecorder {
    state: RefCell<CaptureInputs>,
    rate: u32,
    active: bool,
    shared_timeline: bool,
}

impl PreparedRecorder {
    pub fn failure_evidence(&self, device_ids: &[String]) -> Vec<super::continuity::InputEvidence> {
        self.state
            .borrow()
            .inputs
            .iter()
            .enumerate()
            .map(|(clock_index, input)| super::continuity::InputEvidence {
                clock_index,
                device_selector: input.name.clone(),
                device_id: device_ids[input.routes[0].0].clone(),
                sample_rate_hz: self.rate,
                routes: input
                    .routes
                    .iter()
                    .map(
                        |&(input_index, hardware_channel)| super::continuity::ChannelRoute {
                            input_index,
                            hardware_channel,
                        },
                    )
                    .collect(),
                continuity: input.continuity.snapshot(),
            })
            .collect()
    }

    #[cfg(test)]
    pub fn new(
        inputs: Vec<RunningInput>,
        recordings: Vec<Vec<f32>>,
        rate: u32,
        limit: usize,
    ) -> Self {
        Self::with_storage(inputs, Storage::Memory(recordings), rate, limit)
    }

    pub fn with_storage(
        inputs: Vec<RunningInput>,
        storage: Storage,
        rate: u32,
        limit: usize,
    ) -> Self {
        let shared_timeline = inputs.len() == 1;
        Self {
            state: RefCell::new(CaptureInputs {
                inputs,
                storage,
                limit,
            }),
            rate,
            active: false,
            shared_timeline,
        }
    }
}

impl CaptureInput for PreparedRecorder {
    type Output = CapturedStorage;
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
            hardware_channels: state.storage.channels(),
            sample_rate: self.rate,
        })
    }

    fn input_frame_counts(&self, counts: &mut Vec<usize>) {
        counts.clear();
        let state = self.state.borrow();
        counts.extend((0..state.storage.channels()).map(|channel| state.storage.count(channel)));
    }

    fn check_health(&self) -> Result<(), String> {
        self.state.borrow_mut().drain()
    }

    fn stop(&mut self) -> Result<CapturedStorage, String> {
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
        state.storage.finish(self.rate)
    }

    fn validate(&self, take: &CapturedStorage, rate: u32, channels: usize) -> Result<(), String> {
        match take {
            CapturedStorage::Memory(take) => {
                validate_recorded_take(take, rate, channels, self.shared_timeline)
            }
            CapturedStorage::Files(files) => {
                let state = self.state.borrow();
                if files.len() != channels
                    || files.iter().enumerate().any(|(index, file)| {
                        file.sample_rate_hz != rate
                            || file.samples == 0
                            || file.samples > state.limit
                            || file.samples != state.storage.count(index)
                            || (self.shared_timeline && file.samples != files[0].samples)
                    })
                {
                    return Err("file capture returned inconsistent frame counts or rate".into());
                }
                Ok(())
            }
        }
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
    use std::cell::Cell;
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
            continuity: Arc::new(super::super::continuity::Continuity::default()),
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
    fn file_backend_stops_before_retaining_loss_clock_and_channel_evidence() {
        let root = tempfile::tempdir().unwrap();
        let trace = Rc::new(RefCell::new(Vec::new()));
        let (mut producer, consumer) = rtrb::RingBuffer::new(2);
        let mut prepared = input(0, &trace, false, false);
        prepared.consumer = consumer;
        prepared.routes = vec![(1, 7), (0, 1)];
        super::super::capture_frames_tracked(
            &[0.25_f32; 24],
            8,
            &[7, 1],
            &mut producer,
            &prepared.failed,
            &prepared.continuity,
        );
        let storage = Storage::Files(FileSpool::new(root.path(), 2, 48000).unwrap());
        let mut recorder = PreparedRecorder::with_storage(vec![prepared], storage, 48000, 16);
        recorder.start(None, &[0, 1], 48000, 0.1).unwrap();
        assert!(recorder.check_health().is_err());
        assert!(recorder.stop().is_err());
        assert_eq!(*trace.borrow(), ["play0", "pause0", "drop0"]);
        let evidence = recorder.failure_evidence(&["native-id".into(), "native-id".into()]);
        let decoded = serde_json::to_value(evidence).unwrap();
        assert_eq!(decoded[0]["device_id"], "native-id");
        assert_eq!(decoded[0]["routes"][0]["hardware_channel"], 7);
        assert_eq!(decoded[0]["routes"][0]["input_index"], 1);
        assert_eq!(
            decoded[0]["continuity"]["intervals"][0],
            serde_json::json!({"start_frame":1,"end_frame":3})
        );
        assert_eq!(decoded[0]["continuity"]["missing_frames_per_channel"], 2);
        drop(recorder);
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[test]
    fn independent_ring_lengths_and_origins_survive_the_prepared_backend() {
        for file_backed in [false, true] {
            let root = tempfile::tempdir().unwrap();
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
            let storage = if file_backed {
                Storage::Files(FileSpool::new(root.path(), 2, 48000).unwrap())
            } else {
                Storage::Memory(vec![Vec::new(); 2])
            };
            let mut recorder = PreparedRecorder::with_storage(inputs, storage, 48000, 16);
            recorder.start(None, &[0, 1], 48000, 0.1).unwrap();
            recorder.check_health().unwrap();
            let mut counts = Vec::new();
            recorder.input_frame_counts(&mut counts);
            assert_eq!(counts, [3, 4]);
            assert!(!recorder.shared_timeline);
            producers[0].push(99.0).unwrap();
            let take = recorder.stop().unwrap();
            recorder.validate(&take, 48000, 2).unwrap();
            let recordings = match take {
                CapturedStorage::Memory(take) => take.recorded_per_channel,
                CapturedStorage::Files(files) => files
                    .into_iter()
                    .map(|file| {
                        let mut reader = hound::WavReader::open(file.path()).unwrap();
                        reader
                            .samples::<f32>()
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap()
                    })
                    .collect(),
            };
            assert_eq!(
                recordings,
                [vec![0.0, 1.0, 2.0, 99.0], vec![0.0, 1.0, 2.0, 3.0]]
            );
        }
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

    struct FeedingStream {
        active: Rc<Cell<bool>>,
        stops: Rc<Cell<usize>>,
        cancel: Option<crate::signal_recorder::CancelFlag>,
    }
    impl CaptureStream for FeedingStream {
        fn play(&self) -> Result<(), String> {
            self.active.set(true);
            Ok(())
        }
        fn pause(&self) -> Result<(), String> {
            self.active.set(false);
            self.stops.set(self.stops.get() + 1);
            if let Some(cancel) = &self.cancel {
                cancel.store(true, Ordering::Relaxed);
            }
            Ok(())
        }
    }

    struct FeedingClock {
        now: Rc<Cell<std::time::Duration>>,
        active: Rc<Cell<bool>>,
        producer: rtrb::Producer<f32>,
        failed: Arc<AtomicBool>,
        cancel: crate::signal_recorder::CancelFlag,
        cancel_at: Option<std::time::Duration>,
    }
    impl crate::signal_recorder::acquire::Clock for FeedingClock {
        fn now(&self) -> std::time::Duration {
            self.now.get()
        }
        fn sleep(&mut self, duration: std::time::Duration) {
            self.now.set(self.now.get() + duration);
            if self.active.get() {
                let frames = usize::try_from(duration.as_nanos() * 8000 / 1_000_000_000).unwrap();
                let data: Vec<_> = (0..frames).flat_map(|_| [0.25_f32, -1.25]).collect();
                super::super::capture_frames(&data, 2, &[0, 1], &mut self.producer, &self.failed);
            }
            if self.cancel_at.is_some_and(|at| self.now.get() >= at) {
                self.cancel.store(true, Ordering::Relaxed);
            }
        }
    }

    struct TimedOutput {
        now: Rc<Cell<std::time::Duration>>,
        start: Option<std::time::Duration>,
    }
    impl SweepPlayback for TimedOutput {
        fn start(&mut self, _: &Path, _: Option<&str>, _: u16, _: u32) -> Result<(), String> {
            self.start = Some(self.now.get());
            Ok(())
        }
        fn is_finished(&mut self) -> bool {
            self.now.get() >= self.start.unwrap() + std::time::Duration::from_millis(100)
        }
        fn stop(&mut self) -> Result<(), String> {
            assert!(self.start.take().is_some());
            Ok(())
        }
    }

    fn file_capture(
        cancel_ms: Option<u64>,
        cancel_on_stop: bool,
    ) -> (Result<CapturedStorage, String>, tempfile::TempDir) {
        let root = tempfile::tempdir().unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let now = Rc::new(Cell::new(std::time::Duration::ZERO));
        let active = Rc::new(Cell::new(false));
        let stops = Rc::new(Cell::new(0));
        let failed = Arc::new(AtomicBool::new(false));
        let (producer, consumer) = rtrb::RingBuffer::new(16000);
        let input = RunningInput {
            stream: Some(Box::new(FeedingStream {
                active: Rc::clone(&active),
                stops: Rc::clone(&stops),
                cancel: cancel_on_stop.then(|| Arc::clone(&cancel)),
            })),
            consumer,
            routes: vec![(0, 0), (1, 1)],
            failed: Arc::clone(&failed),
            name: "virtual input".into(),
            continuity: Arc::new(super::super::continuity::Continuity::default()),
        };
        let storage = Storage::Files(FileSpool::new(root.path(), 2, 8000).unwrap());
        let mut recorder = PreparedRecorder::with_storage(vec![input], storage, 8000, 32000);
        let result = crate::signal_recorder::acquire::capture_with_clock(
            &mut TimedOutput {
                now: Rc::clone(&now),
                start: None,
            },
            &mut recorder,
            crate::signal_recorder::acquire::CaptureRequest {
                wav: Path::new("virtual"),
                frames: 800,
                rate: 8000,
                output_channel: 0,
                input_channels: &[0, 1],
                output_device: None,
                input_device: None,
            },
            Some(&cancel),
            &mut FeedingClock {
                now,
                active,
                producer,
                failed,
                cancel: Arc::clone(&cancel),
                cancel_at: cancel_ms.map(std::time::Duration::from_millis),
            },
        );
        assert_eq!(stops.get(), 1, "every started input must stop exactly once");
        (result, root)
    }

    #[test]
    fn file_backend_uses_the_production_lifecycle_and_preserves_input_timeline() {
        let (result, root) = file_capture(None, false);
        let CapturedStorage::Files(files) = result.unwrap() else {
            panic!("expected files")
        };
        assert_eq!(files.len(), 2);
        for (index, file) in files.into_iter().enumerate() {
            assert_eq!(file.samples, 10_800);
            assert_eq!(file.clipped_samples, if index == 0 { 0 } else { 10_800 });
            let target = root.path().join(format!("input-{index}.wav"));
            file.persist(&target).unwrap();
            let mut reader = hound::WavReader::open(target).unwrap();
            assert_eq!(reader.spec().sample_rate, 8000);
            assert!(
                reader
                    .samples::<f32>()
                    .all(|sample| sample.unwrap().to_bits()
                        == if index == 0 { 0.25_f32 } else { -1.25_f32 }.to_bits())
            );
        }
    }

    #[test]
    fn file_backend_cancellation_never_publishes_a_take_and_cleans_all_temporaries() {
        for (at, on_stop) in [
            (Some(100), false),
            (Some(300), false),
            (Some(500), false),
            (Some(1340), false),
            (None, true),
        ] {
            let (result, root) = file_capture(at, on_stop);
            assert_eq!(result.unwrap_err(), "cancelled");
            assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
        }
    }
}
