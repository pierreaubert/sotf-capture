//! Bounded callback loss positions on one native input clock.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
#[cfg(feature = "capture-io")]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[cfg(feature = "capture-io")]
const INTERVALS: usize = 128;

#[cfg(feature = "capture-io")]
#[derive(Debug, Default)]
struct Slot {
    start: AtomicUsize,
    frames: AtomicUsize,
}

/// One callback is the sole writer. Read the snapshot after stopping its stream.
/// Atomics avoid locks, allocation, and logging on both callback entry points.
#[cfg(feature = "capture-io")]
#[derive(Debug)]
pub(super) struct Continuity {
    received: AtomicUsize,
    lost: AtomicUsize,
    events: AtomicUsize,
    invalid: AtomicBool,
    slots: [Slot; INTERVALS],
}

#[cfg(feature = "capture-io")]
impl Default for Continuity {
    fn default() -> Self {
        Self {
            received: AtomicUsize::new(0),
            lost: AtomicUsize::new(0),
            events: AtomicUsize::new(0),
            invalid: AtomicBool::new(false),
            slots: std::array::from_fn(|_| Slot::default()),
        }
    }
}

/// A missing half-open range on one input callback clock.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MissingInterval {
    /// Inclusive position on this input's callback-frame timeline.
    pub start_frame: usize,
    /// Exclusive position, before compacting samples into the ring.
    pub end_frame: usize,
}

/// Bounded discontinuity evidence before samples are compacted into storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Frames delivered by callbacks, including known missing frames.
    pub observed_frames: usize,
    /// Known ring-space losses per channel; invalid data is flagged separately.
    pub missing_frames_per_channel: usize,
    /// Whether malformed data or an input stream error makes continuity uncertain.
    pub invalid_block_or_stream_error: bool,
    /// Loss events beyond the fixed interval capacity; totals still include them.
    pub omitted_intervals: usize,
    /// At most 128 exact gaps, in callback order.
    pub intervals: Vec<MissingInterval>,
}

#[cfg(feature = "capture-io")]
impl Continuity {
    pub fn begin(&self, frames: usize) -> usize {
        self.received.fetch_add(frames, Ordering::Relaxed)
    }

    pub fn invalidate(&self) {
        self.invalid.store(true, Ordering::Relaxed);
    }

    pub fn miss(&self, start: usize, frames: usize) {
        if frames == 0 {
            return;
        }
        self.lost.fetch_add(frames, Ordering::Relaxed);
        let index = self.events.fetch_add(1, Ordering::Relaxed);
        if let Some(slot) = self.slots.get(index) {
            slot.start.store(start, Ordering::Relaxed);
            slot.frames.store(frames, Ordering::Release);
        }
    }

    pub(super) fn snapshot(&self) -> Snapshot {
        let events = self.events.load(Ordering::Acquire);
        let intervals = self.slots[..events.min(INTERVALS)]
            .iter()
            .map(|slot| {
                let frames = slot.frames.load(Ordering::Acquire);
                let start_frame = slot.start.load(Ordering::Relaxed);
                MissingInterval {
                    start_frame,
                    end_frame: start_frame.saturating_add(frames),
                }
            })
            .collect();
        Snapshot {
            observed_frames: self.received.load(Ordering::Relaxed),
            missing_frames_per_channel: self.lost.load(Ordering::Relaxed),
            invalid_block_or_stream_error: self.invalid.load(Ordering::Relaxed),
            omitted_intervals: events.saturating_sub(INTERVALS),
            intervals,
        }
    }
}

/// A logical capture input and its physical hardware channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelRoute {
    /// Zero-based position in the requested input list.
    pub input_index: usize,
    /// Zero-based physical input channel.
    pub hardware_channel: u16,
}

/// Device identity, clock origin, routes, and retained continuity evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputEvidence {
    /// Input clock index within this acquisition; equal indices share one timeline.
    pub clock_index: usize,
    /// User selector used to open the input.
    pub device_selector: String,
    /// Actual native device ID; never substituted with the selector.
    pub device_id: String,
    /// Negotiated nominal input rate.
    pub sample_rate_hz: u32,
    /// Selected channels governed by this snapshot.
    pub routes: Vec<ChannelRoute>,
    /// Input-frame evidence whose origin is the first callback frame.
    pub continuity: Snapshot,
}

#[derive(Serialize)]
pub(super) struct FailureEvidence<'a> {
    pub version: u32,
    pub error: &'a str,
    pub timeline_origin: &'a str,
    pub output_device_id: Option<&'a str>,
    pub output_device_selector: Option<&'a str>,
    pub take_index: Option<usize>,
    pub capture_kind: &'a str,
    pub output_channels: &'a [u16],
    pub output_channel: u16,
    pub stimulus_frames: usize,
    pub inputs: Vec<InputEvidence>,
}

/// Publish a separate failed-acquisition report without overwriting any file.
pub(super) fn save(
    directory: &std::path::Path,
    evidence: &FailureEvidence<'_>,
) -> Result<String, String> {
    use std::io::Write as _;
    let mut temporary = tempfile::Builder::new()
        .prefix(".capture-loss-")
        .suffix(".json")
        .tempfile_in(directory)
        .map_err(|e| format!("cannot stage capture loss evidence: {e}"))?;
    let name = temporary
        .path()
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix('.'))
        .ok_or("invalid loss evidence filename")?
        .to_owned();
    serde_json::to_writer_pretty(&mut temporary, evidence)
        .map_err(|e| format!("cannot serialize capture loss evidence: {e}"))?;
    temporary
        .flush()
        .map_err(|e| format!("cannot flush capture loss evidence: {e}"))?;
    temporary
        .persist_noclobber(directory.join(&name))
        .map_err(|e| format!("cannot publish capture loss evidence: {e}"))?;
    Ok(name)
}

/// Retain failure evidence at an explicit destination, or return it to the caller.
pub(super) fn report_error(
    directory: Option<&std::path::Path>,
    report: &FailureEvidence<'_>,
) -> String {
    if report.inputs.is_empty() {
        return report.error.to_owned();
    }
    let (detail, failed) = match directory {
        Some(directory) => match save(directory, report) {
            Ok(file) => (format!("acquisition evidence: {file}"), false),
            Err(error) => (error, true),
        },
        None => match serde_json::to_string(report) {
            Ok(json) => (format!("transport evidence: {json}"), false),
            Err(error) => (
                format!("cannot serialize transport evidence: {error}"),
                true,
            ),
        },
    };
    if report.error == super::CANCELLED_ERR {
        // Existing sweep/auxiliary callers use exact equality for cancellation.
        // This runs only after streams stop; callbacks never enter the logger.
        if failed {
            log::warn!("Cancelled acquisition: {detail}");
        } else {
            log::debug!("Cancelled acquisition: {detail}");
        }
        report.error.to_owned()
    } else {
        format!("{}; {detail}", report.error)
    }
}

pub(super) fn recording_parent(path: &std::path::Path) -> &std::path::Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."))
}

#[cfg(all(test, feature = "capture-io"))]
#[path = "transport_tests.rs"]
mod transport_tests;

#[cfg(all(test, feature = "capture-io"))]
mod tests {
    use super::*;

    #[test]
    fn capacity_exhaustion_keeps_exact_totals_and_marks_missing_positions() {
        let state = Continuity::default();
        for _ in 0..INTERVALS + 11 {
            let start = state.begin(3);
            state.miss(start + 1, 2);
        }
        let evidence = state.snapshot();
        assert_eq!(evidence.observed_frames, (INTERVALS + 11) * 3);
        assert_eq!(evidence.missing_frames_per_channel, (INTERVALS + 11) * 2);
        assert_eq!(evidence.omitted_intervals, 11);
        assert_eq!(evidence.intervals.len(), INTERVALS);
        assert_eq!(evidence.intervals[127].start_frame, 382);
        assert_eq!(evidence.intervals[127].end_frame, 384);
    }

    #[test]
    fn real_missing_blocks_before_inside_and_after_stimulus_keep_input_positions() {
        // Stimulus occupies [8, 24) on this simulated input clock. No tail-only
        // exception is inferred: every gap must still latch acquisition failure.
        for loss_start in [4, 16, 28] {
            for block in [1, 3, 8] {
                let state = Continuity::default();
                let failed = AtomicBool::new(false);
                let (mut producer, mut consumer) = rtrb::RingBuffer::new(8);
                let mut retained = Vec::new();
                let mut feed = |start: usize, frames: usize| {
                    let data: Vec<f32> = (start..start + frames)
                        .flat_map(|frame| {
                            [
                                f32::from(u16::try_from(frame).unwrap()),
                                -(f32::from(u16::try_from(frame).unwrap())),
                            ]
                        })
                        .collect();
                    super::super::multi_capture::capture_frames_tracked(
                        &data,
                        2,
                        &[1, 0],
                        &mut producer,
                        &failed,
                        &state,
                    );
                };
                // Drain healthy blocks until a four-frame prefix fills the ring.
                for start in (0..loss_start - 4).step_by(4) {
                    feed(start, 4);
                    while let Ok(sample) = consumer.pop() {
                        retained.push(sample);
                    }
                }
                feed(loss_start - 4, 4);
                feed(loss_start, block); // consumer stall discards the actual block
                while let Ok(sample) = consumer.pop() {
                    retained.push(sample);
                }
                for start in (loss_start + block..40).step_by(4) {
                    feed(start, (40 - start).min(4));
                    while let Ok(sample) = consumer.pop() {
                        retained.push(sample);
                    }
                }
                assert!(failed.load(Ordering::Relaxed));
                let expected: Vec<f32> = (0..40)
                    .filter(|frame| !(loss_start..loss_start + block).contains(frame))
                    .flat_map(|frame| {
                        [
                            -(f32::from(u16::try_from(frame).unwrap())),
                            f32::from(u16::try_from(frame).unwrap()),
                        ]
                    })
                    .collect();
                assert_eq!(retained, expected, "loss at {loss_start}, block {block}");
                let snapshot = state.snapshot();
                assert_eq!(snapshot.observed_frames, 40);
                assert_eq!(snapshot.missing_frames_per_channel, block);
                assert_eq!(
                    snapshot.intervals,
                    [MissingInterval {
                        start_frame: loss_start,
                        end_frame: loss_start + block,
                    }]
                );
                assert!(!snapshot.invalid_block_or_stream_error);
            }
        }
    }

    #[test]
    fn invalid_samples_never_publish_a_partial_selected_frame() {
        let state = Continuity::default();
        let failed = AtomicBool::new(false);
        let (mut producer, mut consumer) = rtrb::RingBuffer::new(8);
        super::super::multi_capture::capture_frames_tracked(
            &[0.1_f32, f32::NAN],
            2,
            &[0, 1],
            &mut producer,
            &failed,
            &state,
        );
        assert!(consumer.pop().is_err());
        assert!(failed.load(Ordering::Relaxed));
        let snapshot = state.snapshot();
        assert!(snapshot.invalid_block_or_stream_error);
        assert_eq!(snapshot.observed_frames, 1);
        assert_eq!(snapshot.intervals, []);
    }

    #[test]
    fn saved_failure_keeps_clock_routes_and_exact_loss_without_a_success_artifact() {
        let root = tempfile::tempdir().unwrap();
        let state = Continuity::default();
        state.begin(30);
        state.miss(12, 3);
        let inputs = vec![InputEvidence {
            clock_index: 1,
            device_selector: "second microphone".into(),
            device_id: "actual-native-id".into(),
            sample_rate_hz: 48000,
            routes: vec![
                ChannelRoute {
                    input_index: 2,
                    hardware_channel: 7,
                },
                ChannelRoute {
                    input_index: 3,
                    hardware_channel: 1,
                },
            ],
            continuity: state.snapshot(),
        }];
        let evidence = FailureEvidence {
            version: 1,
            error: "capture overrun",
            timeline_origin: "first input callback",
            output_device_id: Some("actual-output-id"),
            output_device_selector: None,
            take_index: None,
            capture_kind: "session",
            output_channels: &[4],
            output_channel: 4,
            stimulus_frames: 16,
            inputs,
        };
        let first = save(root.path(), &evidence).unwrap();
        let bytes = std::fs::read(root.path().join(&first)).unwrap();
        let second = save(root.path(), &evidence).unwrap();
        assert_ne!(first, second);
        assert_eq!(std::fs::read(root.path().join(first)).unwrap(), bytes);
        let decoded: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded["inputs"][0]["device_id"], "actual-native-id");
        assert_eq!(decoded["inputs"][0]["clock_index"], 1);
        assert_eq!(
            decoded["inputs"][0]["routes"],
            serde_json::json!([{"input_index":2,"hardware_channel":7}, {"input_index":3,"hardware_channel":1}])
        );
        assert_eq!(
            decoded["inputs"][0]["continuity"]["intervals"][0],
            serde_json::json!({"start_frame":12,"end_frame":15})
        );
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 2);
    }
}
