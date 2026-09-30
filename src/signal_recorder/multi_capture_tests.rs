use super::*;

fn request() -> MultiCaptureRequest {
    MultiCaptureRequest {
        inputs: vec![
            CaptureInput {
                device: "aggregate".into(),
                channel: 0,
            },
            CaptureInput {
                device: "aggregate".into(),
                channel: 2,
            },
        ],
        output_device: "DAC".into(),
        output_channel: 1,
        output_overrides: Vec::new(),
        sample_rate_hz: 48_000,
        stimulus: vec![0.0, 0.25, -0.5, 0.0],
    }
}

#[test]
fn aggregate_channels_share_one_stream_but_usb_devices_do_not() {
    let mut request = request();
    let groups = group_inputs(&request).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].routes, vec![(0, 0), (1, 2)]);
    request.inputs[1].device = "second usb mic".into();
    assert_eq!(group_inputs(&request).unwrap().len(), 2);
    request.inputs[1] = request.inputs[0].clone();
    assert!(group_inputs(&request).is_err());
}

#[test]
fn invalid_requests_fail_without_opening_devices() {
    let mut request = request();
    request.stimulus[0] = f32::NAN;
    assert!(group_inputs(&request).is_err());
    request.stimulus[0] = 1.01;
    assert!(group_inputs(&request).is_err());
    request.stimulus.clear();
    assert!(group_inputs(&request).is_err());
}

fn range(
    channels: u16,
    min_rate: u32,
    max_rate: u32,
    format: cpal::SampleFormat,
) -> cpal::SupportedStreamConfigRange {
    cpal::SupportedStreamConfigRange::new(
        channels,
        min_rate,
        max_rate,
        cpal::SupportedBufferSize::Unknown,
        format,
    )
}

#[test]
fn negotiation_requires_exact_rate_and_all_requested_channels() {
    let ranges = vec![
        range(2, 44_100, 44_100, cpal::SampleFormat::F32),
        range(4, 48_000, 96_000, cpal::SampleFormat::I16),
    ];
    let config = exact_config(ranges.clone().into_iter(), 3, 48_000).unwrap();
    assert_eq!(config.sample_rate(), 48_000);
    assert_eq!(config.channels(), 4);
    assert_eq!(config.sample_format(), cpal::SampleFormat::I16);
    assert!(exact_config(ranges.clone().into_iter(), 5, 48_000).is_err());
    assert!(exact_config(ranges.into_iter(), 3, 44_100).is_err());
}

#[test]
fn input_callback_preserves_routing_and_variable_block_boundaries() {
    let (mut producer, mut consumer) = rtrb::RingBuffer::new(4);
    let failed = AtomicBool::new(false);
    capture_frames(&[] as &[f32], 3, &[2, 0], &mut producer, &failed);
    capture_frames(
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        3,
        &[2, 0],
        &mut producer,
        &failed,
    );
    capture_frames(&[0.7, 0.8, 0.9], 3, &[2, 0], &mut producer, &failed);
    assert_eq!(consumer.pop().unwrap(), [0.3, 0.1, 0.0, 0.0]);
    assert_eq!(consumer.pop().unwrap(), [0.6, 0.4, 0.0, 0.0]);
    assert_eq!(consumer.pop().unwrap(), [0.9, 0.7, 0.0, 0.0]);
    assert!(consumer.pop().is_err());
    assert!(!failed.load(Ordering::Relaxed));
}

#[test]
fn input_overflow_nonfinite_and_partial_frames_fail_closed() {
    for data in [&[0.1_f32, 0.2, 0.3, 0.4][..], &[f32::NAN, 0.0], &[0.1]] {
        let (mut producer, _) = rtrb::RingBuffer::new(1);
        let failed = AtomicBool::new(false);
        capture_frames(data, 2, &[0], &mut producer, &failed);
        assert!(failed.load(Ordering::Relaxed));
    }
}

#[test]
fn integer_input_is_converted_without_changing_channel_order() {
    let (mut producer, mut consumer) = rtrb::RingBuffer::new(1);
    let failed = AtomicBool::new(false);
    capture_frames(&[i16::MIN, 0, 16_384], 3, &[2, 0], &mut producer, &failed);
    assert_eq!(consumer.pop().unwrap(), [0.5, -1.0, 0.0, 0.0]);
    assert!(!failed.load(Ordering::Relaxed));
}

#[test]
fn output_callback_zeros_unselected_channels_and_tail() {
    let cursor = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let mut first = [9.0_f32; 6];
    playback_frames(&mut first, 3, 1, &[], &[0.25, -0.5, 0.75], &cursor, &failed);
    assert_eq!(first, [0.0, 0.25, 0.0, 0.0, -0.5, 0.0]);
    let mut second = [9.0_f32; 6];
    playback_frames(
        &mut second,
        3,
        1,
        &[],
        &[0.25, -0.5, 0.75],
        &cursor,
        &failed,
    );
    assert_eq!(second, [0.0, 0.75, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(cursor.load(Ordering::Relaxed), 3);
    assert!(!failed.load(Ordering::Relaxed));
}

#[test]
fn cancellation_prevents_device_access() {
    let cancel = Arc::new(AtomicBool::new(true));
    assert_eq!(
        capture_multidevice(request(), &cancel).unwrap_err(),
        "cancelled"
    );
}

#[test]
fn timing_markers_use_the_fixed_reference_channel_across_callback_boundaries() {
    let overrides = vec![
        CaptureOutputSegment {
            start_frame: 0,
            end_frame: 1,
            channel: 0,
        },
        CaptureOutputSegment {
            start_frame: 3,
            end_frame: 4,
            channel: 0,
        },
    ];
    let stimulus = [0.1, 0.2, 0.3, 0.4];
    let cursor = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let mut first = [9.0_f32; 4];
    playback_frames(&mut first, 2, 1, &overrides, &stimulus, &cursor, &failed);
    assert_eq!(first, [0.1, 0.0, 0.0, 0.2]);
    let mut second = [9.0_f32; 6];
    playback_frames(&mut second, 2, 1, &overrides, &stimulus, &cursor, &failed);
    assert_eq!(second, [0.0, 0.3, 0.4, 0.0, 0.0, 0.0]);
    assert!(!failed.load(Ordering::Relaxed));
}

#[test]
fn overlapping_output_regions_are_rejected_before_audio_access() {
    let mut request = request();
    request.output_overrides = vec![
        CaptureOutputSegment {
            start_frame: 0,
            end_frame: 3,
            channel: 0,
        },
        CaptureOutputSegment {
            start_frame: 2,
            end_frame: 4,
            channel: 1,
        },
    ];
    assert!(group_inputs(&request).is_err());
}
