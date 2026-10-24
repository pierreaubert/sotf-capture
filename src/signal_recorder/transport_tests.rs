//! Real callback gaps and their recording/error provenance, without devices.

// Rust guideline compliant 2026-02-21

use super::{
    ChannelRoute, Continuity, FailureEvidence, InputEvidence, MissingInterval, report_error,
};
use crate::signal_recorder::{self as recorder, SweepRecorder};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

fn sample(frame: usize) -> f32 {
    f32::from(u16::try_from(frame).unwrap()) * 0.001
}

fn packets(start: usize, missing: usize) -> Vec<(usize, usize, bool)> {
    let mut packets: Vec<_> = (0..start - 4)
        .step_by(4)
        .map(|first| (first, 4, true))
        .collect();
    packets.push((start - 4, 4, false));
    packets.push((start, missing, true));
    packets.extend(
        (start + missing..40)
            .step_by(4)
            .map(|first| (first, (40 - first).min(4), true)),
    );
    packets
}

fn evidence(clock: &Continuity, routes: Vec<ChannelRoute>) -> InputEvidence {
    InputEvidence {
        clock_index: 0,
        device_selector: "synthetic input".into(),
        device_id: "synthetic-native-id".into(),
        sample_rate_hz: 48_000,
        routes,
        continuity: clock.snapshot(),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "One fixture compares both production input cores and their exact retained samples and intervals."
)]
fn missing_audio(
    auxiliary: bool,
    start: usize,
    missing: usize,
) -> (Vec<Vec<f32>>, Vec<InputEvidence>) {
    let mut retained = vec![Vec::new(), Vec::new()];
    let failed = AtomicBool::new(false);
    let lost = AtomicUsize::new(0);
    let reports = if auxiliary {
        let (mut producer, mut consumer) = rtrb::RingBuffer::new(4);
        let accepted = AtomicUsize::new(0);
        let clock = Continuity::default();
        for (first, frames, drain) in packets(start, missing) {
            let data: Vec<_> = (first..first + frames)
                .flat_map(|frame| [sample(frame), -sample(frame)])
                .collect();
            recorder::build::capture_pairs(
                &data,
                2,
                1,
                Some(0),
                &mut producer,
                recorder::build::InputCounters {
                    accepted: &accepted,
                    lost: &lost,
                    failed: &failed,
                    continuity: &clock,
                },
            );
            if drain {
                while let Ok((mic, loopback)) = consumer.pop() {
                    retained[0].push(mic);
                    retained[1].push(loopback);
                }
            }
        }
        assert_eq!(lost.load(Ordering::Relaxed), missing);
        vec![evidence(
            &clock,
            vec![
                ChannelRoute {
                    input_index: 0,
                    hardware_channel: 1,
                },
                ChannelRoute {
                    input_index: 1,
                    hardware_channel: 0,
                },
            ],
        )]
    } else {
        let (mut producers, mut consumers): (Vec<_>, Vec<_>) =
            (0..2).map(|_| rtrb::RingBuffer::new(4)).unzip();
        let counts: Vec<_> = (0..2).map(|_| Arc::new(AtomicUsize::new(0))).collect();
        let clocks: Vec<_> = (0..2).map(|_| Arc::new(Continuity::default())).collect();
        for (first, frames, drain) in packets(start, missing) {
            let data: Vec<_> = (first..first + frames)
                .flat_map(|frame| [sample(frame), -sample(frame)])
                .collect();
            recorder::recorder::capture_channels(
                &data,
                2,
                &mut recorder::recorder::InputRings {
                    producers: &mut producers,
                    counts: &counts,
                    channels: &[1, 0],
                    overruns: &lost,
                    failed: &failed,
                    meter: None,
                    continuity: &clocks,
                },
            );
            if drain {
                for (index, consumer) in consumers.iter_mut().enumerate() {
                    while let Ok(value) = consumer.pop() {
                        retained[index].push(value);
                    }
                }
            }
        }
        assert_eq!(lost.load(Ordering::Relaxed), missing * 2);
        clocks
            .iter()
            .enumerate()
            .map(|(index, clock)| {
                evidence(
                    clock,
                    vec![ChannelRoute {
                        input_index: index,
                        hardware_channel: u16::try_from(1 - index).unwrap(),
                    }],
                )
            })
            .collect()
    };
    assert!(!failed.load(Ordering::Relaxed)); // aggregate loss independently rejects acquisition
    let expected: Vec<_> = (0..40)
        .filter(|frame| !(start..start + missing).contains(frame))
        .map(sample)
        .collect();
    assert_eq!(retained[1], expected);
    assert_eq!(
        retained[0],
        expected.iter().map(|value| -value).collect::<Vec<_>>()
    );
    for report in &reports {
        assert_eq!(report.continuity.observed_frames, 40);
        assert_eq!(report.continuity.missing_frames_per_channel, missing);
        assert_eq!(
            report.continuity.intervals,
            [MissingInterval {
                start_frame: start,
                end_frame: start + missing
            }]
        );
        assert_eq!(report.continuity.omitted_intervals, 0);
    }
    (retained, reports)
}

#[test]
fn invalid_blocks_do_not_shift_subsequent_known_ring_gap_positions() {
    for auxiliary in [false, true] {
        let clock = Arc::new(Continuity::default());
        let accepted = Arc::new(AtomicUsize::new(0));
        let lost = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let invalid = [f32::NAN, 0.1, 0.2, 0.3];
        let valid = [0.25_f32; 6];
        if auxiliary {
            let (mut producer, _consumer) = rtrb::RingBuffer::new(1);
            for data in [&invalid[..], &valid[..]] {
                recorder::build::capture_pairs(
                    data,
                    2,
                    0,
                    None,
                    &mut producer,
                    recorder::build::InputCounters {
                        accepted: &accepted,
                        lost: &lost,
                        failed: &failed,
                        continuity: &clock,
                    },
                );
            }
        } else {
            let (producer, _consumer) = rtrb::RingBuffer::new(1);
            let mut producers = [producer];
            for data in [&invalid[..], &valid[..]] {
                recorder::recorder::capture_channels(
                    data,
                    2,
                    &mut recorder::recorder::InputRings {
                        producers: &mut producers,
                        counts: &[Arc::clone(&accepted)],
                        channels: &[0],
                        overruns: &lost,
                        failed: &failed,
                        meter: None,
                        continuity: &[Arc::clone(&clock)],
                    },
                );
            }
        }
        let evidence = clock.snapshot();
        assert!(evidence.invalid_block_or_stream_error);
        assert_eq!(evidence.observed_frames, 5);
        assert_eq!(evidence.missing_frames_per_channel, 2);
        assert_eq!(
            evidence.intervals,
            [MissingInterval {
                start_frame: 3,
                end_frame: 5
            }]
        );
    }
}

#[test]
fn sweep_and_auxiliary_retain_real_missing_blocks_before_inside_and_after_stimulus() {
    // Simulated stimulus lies in [8, 24). Every gap is invalid, including tail loss.
    for auxiliary in [false, true] {
        for start in [4, 16, 28] {
            for missing in [1, 3, 8] {
                missing_audio(auxiliary, start, missing);
            }
        }
    }
}

struct LossRecorder {
    inner: recorder::FakeRecorder,
    starts: usize,
    channels: usize,
    stopped: bool,
    samples: Vec<Vec<f32>>,
    inputs: Vec<InputEvidence>,
}

impl SweepRecorder for LossRecorder {
    fn start(
        &mut self,
        device: Option<&str>,
        channels: &[u16],
        rate: u32,
        duration: f64,
    ) -> Result<recorder::NegotiatedInput, String> {
        self.stopped = false;
        let config = self.inner.start(device, channels, rate, duration)?;
        self.channels = channels.len();
        self.starts += 1;
        Ok(config)
    }
    fn captured_samples(&self) -> usize {
        self.inner.captured_samples()
    }
    fn stop(&mut self) -> Result<recorder::RecordedTake, String> {
        let mut take = self.inner.stop()?;
        if self.starts > 1 {
            take.recorded_per_channel = self.samples[..self.channels].to_vec();
            take.dropped_samples = u64::try_from(3 * self.channels).unwrap();
            self.stopped = true;
        }
        Ok(take)
    }
    fn transport_evidence(&self) -> Vec<InputEvidence> {
        if self.stopped {
            self.inputs[..self.channels].to_vec()
        } else {
            Vec::new()
        }
    }
}

#[test]
fn public_sweep_apis_save_loss_after_stop_and_never_publish_analysis() {
    for multi in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let (samples, inputs) = missing_audio(false, 16, 3);
        let wire = recorder::VirtualWire::default();
        let mut playback = wire.playback("loss test");
        let mut input = LossRecorder {
            inner: wire.recorder("loss test"),
            starts: 0,
            channels: 0,
            stopped: false,
            samples,
            inputs,
        };
        let reference = vec![0.25_f32; 4800];
        let stimulus = recorder::write_temp_wav(&reference, 48_000, 1).unwrap();
        let wav = root.path().join("raw.wav");
        let csv = root.path().join("analysis.csv");
        let error = if multi {
            recorder::record_and_analyze_with_multi_recorder(
                &mut playback,
                &mut input,
                stimulus.path(),
                &[wav.clone(), root.path().join("raw-2.wav")],
                &reference,
                48_000,
                &[csv.clone(), root.path().join("analysis-2.csv")],
                0,
                &[1, 0],
                None,
                None,
                &[None, None],
                None,
                1,
                None,
            )
            .unwrap_err()
        } else {
            recorder::record_and_analyze_with_recorder(
                &mut playback,
                &mut input,
                stimulus.path(),
                &wav,
                &reference,
                48_000,
                &csv,
                0,
                1,
                None,
                None,
                None,
                None,
                1,
                None,
            )
            .unwrap_err()
        };
        assert!(input.stopped);
        assert!(error.contains("capture lost"), "{error}");
        assert!(!wav.exists());
        assert!(!csv.exists());
        let files: Vec<_> = std::fs::read_dir(root.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(files.len(), 1);
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
        assert_eq!(json["capture_kind"], "sweep");
        assert_eq!(json["take_index"], 0);
        assert_eq!(json["inputs"][0]["routes"][0]["hardware_channel"], 1);
        assert_eq!(
            json["inputs"][0]["continuity"]["intervals"][0]["start_frame"],
            16
        );
        assert!(error.contains(files[0].file_name().unwrap().to_str().unwrap()));
    }
}

#[test]
fn auxiliary_reports_save_or_return_loss_and_preserve_exact_cancellation() {
    let root = tempfile::tempdir().unwrap();
    let (_, inputs) = missing_audio(true, 28, 8);
    let mut report = FailureEvidence {
        version: 1,
        error: "auxiliary capture lost 8 input frames",
        timeline_origin: "first input callback",
        output_device_id: Some("synthetic-output"),
        output_device_selector: None,
        take_index: None,
        capture_kind: "bass anchor",
        output_channels: &[0, 2],
        output_channel: 0,
        stimulus_frames: 16,
        inputs,
    };
    let error = report_error(Some(root.path()), &report);
    assert!(error.contains("acquisition evidence: capture-loss-"));
    let inline = report_error(None, &report);
    let (_, json) = inline.split_once("; transport evidence: ").unwrap();
    let decoded: serde_json::Value = serde_json::from_str(json).unwrap();
    assert_eq!(decoded["inputs"][0]["routes"].as_array().unwrap().len(), 2);
    assert_eq!(
        decoded["inputs"][0]["continuity"]["intervals"][0],
        serde_json::json!({"start_frame":28,"end_frame":36})
    );
    report.error = recorder::CANCELLED_ERR;
    assert_eq!(
        report_error(Some(root.path()), &report),
        recorder::CANCELLED_ERR
    );
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 2);
    report.error = "capture failed";
    let failed = report_error(Some(&root.path().join("missing-directory")), &report);
    assert!(failed.contains("cannot stage capture loss evidence"));
}
