//! Simultaneous, independently clocked measurement input streams.
//!
//! This blocking control-thread API captures raw samples only. Opening every
//! input before playback prevents sequential recordings from being mistaken for
//! simultaneous takes. Acoustic chirps, clock correction and uncertainty belong
//! to the caller's measurement protocol; equal nominal rates do not imply sync.

use cpal::traits::DeviceTrait;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub(super) use super::continuity;
mod lifecycle;
mod spool;

pub use spool::CapturedFile;

use super::CancelFlag;
use super::measurement::measurement_sample_format_rank;
// Protocol types live in `super::types` so DSP-only builds (no `capture-io`)
// keep them; re-exported here so the pre-existing paths keep working.
pub use super::types::{
    CaptureInput, CaptureOutputSegment, MultiCaptureRequest, MultiCaptureResult,
};

// Thirty-two microphones plus one electrical loopback is the capture limit.
// Callbacks copy selected channels into preallocated interleaved rings.
const MAX_INPUTS: usize = 33;
// Bound each input; the shared aggregate byte budget also covers all rings.
const MAX_FRAMES: usize = 16_000_000;

#[derive(Debug)]
struct InputGroup {
    name: String,
    routes: Vec<(usize, u16)>,
}

fn route_groups(request: &MultiCaptureRequest) -> Result<Vec<InputGroup>, String> {
    if !(2..=MAX_INPUTS).contains(&request.inputs.len()) {
        return Err("simultaneous capture requires two to thirty-three inputs (up to thirty-two microphones plus loopback)".into());
    }
    if !(16_001..=384_000).contains(&request.sample_rate_hz)
        || request.stimulus.is_empty()
        || request.stimulus.len() > MAX_FRAMES.saturating_sub(request.sample_rate_hz as usize * 10)
        || request
            .stimulus
            .iter()
            .any(|s| !s.is_finite() || s.abs() > 1.0)
        || request.output_device.trim().is_empty()
        || request.output_channel >= 64
    {
        return Err("invalid capture rate, stimulus, duration or output route".into());
    }
    let mut previous_end = 0;
    if request.output_overrides.len() > 16 {
        return Err("too many capture output regions".into());
    }
    for region in &request.output_overrides {
        if region.channel >= 64
            || region.start_frame < previous_end
            || region.start_frame >= region.end_frame
            || region.end_frame > request.stimulus.len()
        {
            return Err("invalid or overlapping capture output regions".into());
        }
        previous_end = region.end_frame;
    }
    let mut groups: Vec<InputGroup> = Vec::new();
    for (index, input) in request.inputs.iter().enumerate() {
        if input.device.trim().is_empty() || input.channel >= 64 {
            return Err("input device must be explicit and channel below 64".into());
        }
        if let Some(group) = groups.iter_mut().find(|group| group.name == input.device) {
            if group
                .routes
                .iter()
                .any(|(_, channel)| *channel == input.channel)
            {
                return Err("duplicate microphone device/channel route".into());
            }
            group.routes.push((index, input.channel));
        } else {
            groups.push(InputGroup {
                name: input.device.clone(),
                routes: vec![(index, input.channel)],
            });
        }
    }
    Ok(groups)
}

fn group_inputs(request: &MultiCaptureRequest) -> Result<Vec<InputGroup>, String> {
    let groups = route_groups(request)?;
    crate::limits::capture_bytes(
        request.inputs.len(),
        request.stimulus.len() + request.sample_rate_hz as usize * 10,
        request.sample_rate_hz,
        request.stimulus.len(),
    )?;
    Ok(groups)
}

fn exact_config(
    supported: impl Iterator<Item = cpal::SupportedStreamConfigRange>,
    required_channels: u16,
    rate: u32,
) -> Result<cpal::SupportedStreamConfig, String> {
    supported
        // Selected routes are bounded separately. Interfaces such as Fireface
        // expose wider native streams even when only low-numbered routes are used.
        .filter(|c| c.channels() >= required_channels
            && c.min_sample_rate() <= rate && c.max_sample_rate() >= rate
            && measurement_sample_format_rank(c.sample_format()) < 12)
        .min_by_key(|c| (measurement_sample_format_rank(c.sample_format()), c.channels()))
        .map(|c| c.with_sample_rate(rate))
        .ok_or_else(|| format!("device cannot capture/play {required_channels} channels at exactly {rate} Hz in a supported PCM format"))
}

// Shared pure callback cores are also exercised without opening devices.
pub(super) struct CaptureWrite {
    frames: usize,
    invalid: bool,
}

pub(super) fn capture_frames<T>(
    data: &[T],
    channels: usize,
    selected: &[u16],
    producer: &mut rtrb::Producer<f32>,
    failed: &AtomicBool,
) -> CaptureWrite
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    if channels == 0
        || !data.len().is_multiple_of(channels)
        || selected.is_empty()
        || selected.len() > MAX_INPUTS
        || selected
            .iter()
            .any(|channel| usize::from(*channel) >= channels)
    {
        failed.store(true, Ordering::Relaxed);
        return CaptureWrite {
            frames: 0,
            invalid: true,
        };
    }
    for (index, frame) in data.chunks_exact(channels).enumerate() {
        if selected.iter().any(|channel| {
            !<f32 as cpal::FromSample<T>>::from_sample_(frame[usize::from(*channel)]).is_finite()
        }) {
            failed.store(true, Ordering::Relaxed);
            return CaptureWrite {
                frames: index,
                invalid: true,
            };
        }
        // The single producer owns these slots; the consumer can only free
        // more. Reserve a whole frame logically before publishing its samples.
        if producer.slots() < selected.len() {
            failed.store(true, Ordering::Relaxed);
            return CaptureWrite {
                frames: index,
                invalid: false,
            };
        }
        for channel in selected {
            let sample = <f32 as cpal::FromSample<T>>::from_sample_(frame[usize::from(*channel)]);
            if producer.push(sample).is_err() {
                failed.store(true, Ordering::Relaxed);
                return CaptureWrite {
                    frames: index,
                    invalid: true,
                };
            }
        }
    }
    CaptureWrite {
        frames: data.len() / channels,
        invalid: false,
    }
}

pub(super) fn capture_frames_tracked<T>(
    data: &[T],
    channels: usize,
    selected: &[u16],
    producer: &mut rtrb::Producer<f32>,
    failed: &AtomicBool,
    continuity: &continuity::Continuity,
) where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let frames = data.len().checked_div(channels).unwrap_or(0);
    let start = continuity.begin(frames);
    let written = capture_frames(data, channels, selected, producer, failed);
    if written.invalid {
        // The device did not provide a usable block. Do not invent an exact
        // loss interval for malformed data or an unexpected ring failure.
        continuity.invalidate();
    } else if written.frames < frames {
        continuity.miss(start + written.frames, frames - written.frames);
    }
}

pub(super) fn playback_frames<T>(
    data: &mut [T],
    channels: usize,
    selected: usize,
    overrides: &[CaptureOutputSegment],
    stimulus: &[f32],
    cursor: &AtomicUsize,
    failed: &AtomicBool,
) where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    data.fill(T::from_sample_(0.0));
    if channels == 0
        || selected >= channels
        || !data.len().is_multiple_of(channels)
        || overrides.len() > 16
        || overrides
            .iter()
            .any(|region| usize::from(region.channel) >= channels)
    {
        failed.store(true, Ordering::Relaxed);
        return;
    }
    let mut position = cursor.load(Ordering::Relaxed);
    for frame in data.chunks_exact_mut(channels) {
        if let Some(sample) = stimulus.get(position) {
            let channel = overrides
                .iter()
                .find(|region| position >= region.start_frame && position < region.end_frame)
                .map_or(selected, |region| usize::from(region.channel));
            frame[channel] = T::from_sample_(sample.clamp(-1.0, 1.0));
            position += 1;
        }
    }
    cursor.store(position, Ordering::Release);
}

fn input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    selected: Vec<u16>,
    mut producer: rtrb::Producer<f32>,
    failed: Arc<AtomicBool>,
    continuity: Arc<continuity::Continuity>,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = usize::from(config.channels);
    let data_failed = Arc::clone(&failed);
    let data_continuity = Arc::clone(&continuity);
    device
        .build_input_stream::<T, _, _>(
            config,
            move |data, _| {
                capture_frames_tracked(
                    data,
                    channels,
                    &selected,
                    &mut producer,
                    &data_failed,
                    &data_continuity,
                );
            },
            move |_| {
                failed.store(true, Ordering::Relaxed);
                continuity.invalidate();
            },
            None,
        )
        .map_err(|e| format!("cannot build capture input: {e}"))
}

#[expect(
    clippy::too_many_arguments,
    reason = "Pass the preallocated callback state and its independent presentation/error signals explicitly."
)]
fn output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    selected: usize,
    overrides: Vec<CaptureOutputSegment>,
    stimulus: Arc<Vec<f32>>,
    cursor: Arc<AtomicUsize>,
    failed: Arc<AtomicBool>,
    presentation: Arc<super::presentation::PresentationClock>,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let channels = usize::from(config.channels);
    let data_failed = Arc::clone(&failed);
    let rate = config.sample_rate;
    device
        .build_output_stream::<T, _, _>(
            config,
            move |data, info| {
                let first = cursor.load(Ordering::Relaxed);
                playback_frames(
                    data,
                    channels,
                    selected,
                    &overrides,
                    &stimulus,
                    &cursor,
                    &data_failed,
                );
                if first < stimulus.len() && cursor.load(Ordering::Acquire) >= stimulus.len() {
                    presentation.submit(info, stimulus.len() - first, rate, &data_failed);
                }
            },
            move |_| {
                failed.store(true, Ordering::Relaxed);
            },
            None,
        )
        .map_err(|e| format!("cannot build capture output: {e}"))
}

macro_rules! with_sample_format {
    ($format:expr, $function:ident, $($arg:expr),+ $(,)?) => {
        match $format {
            cpal::SampleFormat::F32 => $function::<f32>($($arg),+),
            cpal::SampleFormat::F64 => $function::<f64>($($arg),+),
            cpal::SampleFormat::I8 => $function::<i8>($($arg),+),
            cpal::SampleFormat::I16 => $function::<i16>($($arg),+),
            cpal::SampleFormat::I24 => $function::<cpal::I24>($($arg),+),
            cpal::SampleFormat::I32 => $function::<i32>($($arg),+),
            cpal::SampleFormat::I64 => $function::<i64>($($arg),+),
            cpal::SampleFormat::U8 => $function::<u8>($($arg),+),
            cpal::SampleFormat::U16 => $function::<u16>($($arg),+),
            cpal::SampleFormat::U24 => $function::<cpal::U24>($($arg),+),
            cpal::SampleFormat::U32 => $function::<u32>($($arg),+),
            cpal::SampleFormat::U64 => $function::<u64>($($arg),+),
            format => Err(format!("unsupported capture sample format {format:?}")),
        }
    }
}

pub(super) fn allocate_recordings(inputs: usize, limit: usize) -> Result<Vec<Vec<f32>>, String> {
    let mut recordings = Vec::with_capacity(inputs);
    for _ in 0..inputs {
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(limit)
            .map_err(|error| format!("cannot allocate capture storage: {error}"))?;
        recordings.push(samples);
    }
    Ok(recordings)
}

pub(super) fn input_ring(rate: u32, selected: usize) -> (rtrb::Producer<f32>, rtrb::Consumer<f32>) {
    rtrb::RingBuffer::new(rate as usize * selected)
}

/// Accelerate a hardware-free producer while keeping the real bounded writer.
#[cfg(feature = "test-util")]
pub(crate) fn probe_file_spool(
    root: &std::path::Path,
    inputs: usize,
    rate: u32,
    frames: usize,
    sample_live_storage: impl FnOnce(),
) -> Result<(), String> {
    let mut spool = spool::FileSpool::new(root, inputs, rate)?;
    for first in (0..frames).step_by(crate::limits::FILE_CHUNK_FRAMES) {
        let count = (frames - first).min(crate::limits::FILE_CHUNK_FRAMES);
        for channel in 0..inputs {
            for _ in 0..count {
                spool.push(channel, 0.25)?;
            }
        }
        // This pacing belongs to the hardware-free probe only. Native capture
        // never waits for disk here; a full queue fails that acquisition.
        spool.settle_for_probe()?;
    }
    sample_live_storage();
    let files = spool.finish()?;
    if files.len() != inputs
        || files
            .iter()
            .any(|file| file.samples != frames || file.sample_rate_hz != rate)
    {
        return Err("file storage probe produced inconsistent WAV lengths".into());
    }
    for file in &files {
        let mut reader = hound::WavReader::open(file.path()).map_err(|error| error.to_string())?;
        if usize::try_from(reader.duration()).map_err(|error| error.to_string())? != frames
            || reader.spec().sample_rate != rate
        {
            return Err("file storage probe WAV header disagrees with frame evidence".into());
        }
        for sample in reader.samples::<f32>() {
            if sample.map_err(|error| error.to_string())?.to_bits() != 0.25_f32.to_bits() {
                return Err("file storage probe WAV payload changed".into());
            }
        }
    }
    Ok(())
}

struct RunningInput {
    stream: Option<Box<dyn lifecycle::CaptureStream>>,
    consumer: rtrb::Consumer<f32>,
    routes: Vec<(usize, u16)>,
    failed: Arc<AtomicBool>,
    name: String,
    continuity: Arc<continuity::Continuity>,
}

fn drain_storage(
    inputs: &mut [RunningInput],
    storage: &mut lifecycle::Storage,
    limit: usize,
) -> Result<(), String> {
    for input in inputs {
        if input.failed.load(Ordering::Relaxed) {
            return Err(format!(
                "capture failed on {}: stream error, malformed samples, or buffer overrun",
                input.name
            ));
        }
        // Drain only a bounded snapshot. A running producer must not keep this
        // control loop busy forever and prevent cancellation/other-device polling.
        for _ in 0..input.consumer.slots() / input.routes.len() {
            for (index, _) in &input.routes {
                if storage.count(*index) >= limit {
                    return Err("capture exceeded bounded recording storage".into());
                }
                let sample = input
                    .consumer
                    .pop()
                    .map_err(|_| "capture frame became incomplete")?;
                storage.push(*index, sample)?;
            }
        }
    }
    Ok(())
}

/// Capture every microphone while playing one source's mono stimulus.
///
/// Call on a worker thread. Inputs on one aggregate device share a stream.
/// Independent devices have independent sample origins; returned arrays must not
/// be treated as aligned. Callbacks use preallocated rings and atomics only.
/// Cancellation and failures drop every stream before returning.
///
/// # Errors
/// Rejects invalid routes, ambiguous names, rate/format mismatches, stream errors,
/// malformed or dropped frames, cancellation, storage limits and device stalls.
pub fn capture_multidevice(
    request: MultiCaptureRequest,
    cancel: &CancelFlag,
) -> Result<MultiCaptureResult, String> {
    let captured = capture_native(request, cancel, None)?;
    let lifecycle::CapturedStorage::Memory(take) = captured.storage else {
        unreachable!("memory capture selected file storage")
    };
    Ok(MultiCaptureResult {
        input_device_ids: captured.input_device_ids,
        output_device_id: captured.output_device_id,
        sample_rate_hz: take.analysis_sample_rate,
        recordings: take.recorded_per_channel,
        input_sample_formats: captured.input_sample_formats,
        output_sample_format: captured.output_sample_format,
    })
}

/// Raw capture metadata and finalized private WAVs in request order.
#[derive(Debug)]
pub struct FileCaptureResult {
    /// Actual input device identities, in request order.
    pub input_device_ids: Vec<String>,
    /// Actual output device identity.
    pub output_device_id: String,
    /// Confirmed nominal device rate.
    pub sample_rate_hz: u32,
    /// Finalized mono WAVs; independent inputs retain their own sample origins.
    pub recordings: Vec<CapturedFile>,
    /// Input PCM representations in request order.
    pub input_sample_formats: Vec<String>,
    /// Output PCM representation.
    pub output_sample_format: String,
}

/// Capture through bounded rings and a disk writer without retaining whole takes.
///
/// `directory` must exist. Files remain private temporary artifacts until the
/// caller explicitly publishes them; failed/cancelled captures remove them.
/// A lifecycle failure also saves a separate `capture-loss-*.json` report after
/// streams stop. The error names that report, or explains why it could not be
/// saved. Its positions belong to individual input clocks, not stimulus offsets.
///
/// # Errors
/// Returns an error for invalid requests, device/transport failures, cancellation,
/// storage limits, slow disks, writer failures, or finalization failures.
pub fn capture_multidevice_to_files(
    request: MultiCaptureRequest,
    directory: &std::path::Path,
    cancel: &CancelFlag,
) -> Result<FileCaptureResult, String> {
    let captured = capture_native(request, cancel, Some(directory))?;
    let lifecycle::CapturedStorage::Files(files) = captured.storage else {
        unreachable!("file capture selected memory storage")
    };
    Ok(FileCaptureResult {
        input_device_ids: captured.input_device_ids,
        output_device_id: captured.output_device_id,
        sample_rate_hz: captured.rate,
        recordings: files,
        input_sample_formats: captured.input_sample_formats,
        output_sample_format: captured.output_sample_format,
    })
}

struct NativeCapture {
    input_device_ids: Vec<String>,
    output_device_id: String,
    rate: u32,
    input_sample_formats: Vec<String>,
    output_sample_format: String,
    storage: lifecycle::CapturedStorage,
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep native stream preparation and lifecycle cleanup in one ordered operation."
)]
fn capture_native(
    request: MultiCaptureRequest,
    cancel: &CancelFlag,
    directory: Option<&std::path::Path>,
) -> Result<NativeCapture, String> {
    let groups = if directory.is_some() {
        let groups = route_groups(&request)?;
        crate::limits::file_capture_bytes(
            request.inputs.len(),
            request.sample_rate_hz,
            request.stimulus.len(),
        )?;
        groups
    } else {
        group_inputs(&request)?
    };
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    let host = crate::devices::get_host_for_device(Some(&request.output_device));
    let output = crate::devices::find_device(&host, &request.output_device, false)?;
    let output_config = exact_config(
        output
            .supported_output_configs()
            .map_err(|e| e.to_string())?,
        request
            .output_overrides
            .iter()
            .map(|region| region.channel)
            .chain(std::iter::once(request.output_channel))
            .max()
            .unwrap_or(request.output_channel)
            + 1,
        request.sample_rate_hz,
    )?;
    let output_sample_format = format!("{:?}", output_config.sample_format());
    let output_device_id = output
        .id()
        .map_err(|e| format!("cannot identify playback device: {e}"))?
        .to_string();
    let mut inputs = Vec::with_capacity(groups.len());
    let mut input_sample_formats = vec![String::new(); request.inputs.len()];
    let mut input_device_ids = vec![String::new(); request.inputs.len()];
    let limit = request.stimulus.len() + request.sample_rate_hz as usize * 10;
    let storage = if let Some(directory) = directory {
        lifecycle::Storage::Files(spool::FileSpool::new(
            directory,
            request.inputs.len(),
            request.sample_rate_hz,
        )?)
    } else {
        lifecycle::Storage::Memory(allocate_recordings(request.inputs.len(), limit)?)
    };
    // Build every stream before any stream starts. Dropping this vector handles
    // cleanup if device setup or any later play() call fails partway through.
    for group in groups {
        let input_host = crate::devices::get_host_for_device(Some(&group.name));
        let device = crate::devices::find_device(&input_host, &group.name, true)?;
        let device_id = device
            .id()
            .map_err(|e| format!("cannot identify microphone device: {e}"))?
            .to_string();
        // Different selectors can refer to the same physical device. Refuse a
        // second stream rather than treating aliases as independent USB clocks.
        if input_device_ids.iter().any(|id| id == &device_id) {
            return Err("microphones on the same device must use the same device selector".into());
        }
        let channels = group
            .routes
            .iter()
            .map(|(_, channel)| channel + 1)
            .max()
            .unwrap_or(1);
        let config = exact_config(
            device
                .supported_input_configs()
                .map_err(|e| e.to_string())?,
            channels,
            request.sample_rate_hz,
        )
        .map_err(|e| format!("{}: {e}", group.name))?;
        for (index, _) in &group.routes {
            input_sample_formats[*index] = format!("{:?}", config.sample_format());
            input_device_ids[*index].clone_from(&device_id);
        }
        let (producer, consumer) = input_ring(request.sample_rate_hz, group.routes.len());
        let failed = Arc::new(AtomicBool::new(false));
        let continuity = Arc::new(continuity::Continuity::default());
        let selected = group.routes.iter().map(|(_, channel)| *channel).collect();
        let stream = with_sample_format!(
            config.sample_format(),
            input_stream,
            &device,
            &config.config(),
            selected,
            producer,
            Arc::clone(&failed),
            Arc::clone(&continuity)
        )?;
        inputs.push(RunningInput {
            stream: Some(Box::new(stream)),
            consumer,
            routes: group.routes,
            failed,
            name: group.name,
            continuity,
        });
    }
    let stimulus = Arc::new(request.stimulus);
    let cursor = Arc::new(AtomicUsize::new(0));
    let output_failed = Arc::new(AtomicBool::new(false));
    let presentation = Arc::new(super::presentation::PresentationClock::new());
    let output_stream = with_sample_format!(
        output_config.sample_format(),
        output_stream,
        &output,
        &output_config.config(),
        usize::from(request.output_channel),
        request.output_overrides,
        Arc::clone(&stimulus),
        Arc::clone(&cursor),
        Arc::clone(&output_failed),
        Arc::clone(&presentation)
    )?;
    let mut recorder =
        lifecycle::PreparedRecorder::with_storage(inputs, storage, request.sample_rate_hz, limit);
    let mut playback =
        lifecycle::PreparedPlayback::new(Box::new(output_stream), output_failed, presentation);
    let logical_channels: Vec<_> = (0..request.inputs.len())
        .map(|index| u16::try_from(index).map_err(|_| "input channel count overflow"))
        .collect::<Result<_, _>>()?;
    let captured = super::acquire::capture(
        &mut playback,
        &mut recorder,
        super::acquire::CaptureRequest {
            wav: std::path::Path::new("prepared-session-stimulus"),
            frames: stimulus.len(),
            rate: request.sample_rate_hz,
            output_channel: request.output_channel,
            input_channels: &logical_channels,
            output_device: None,
            input_device: None,
        },
        Some(cancel),
    );
    let captured = match captured {
        Ok(captured) => captured,
        Err(error) => {
            let evidence = continuity::FailureEvidence {
                version: 1,
                error: &error,
                timeline_origin: "frame zero is the first callback frame of each input stream; clocks are independent; positions are not stimulus offsets",
                output_device_id: Some(&output_device_id),
                output_device_selector: None,
                take_index: None,
                capture_kind: "session",
                output_channels: &[request.output_channel],
                output_channel: request.output_channel,
                stimulus_frames: stimulus.len(),
                inputs: recorder.failure_evidence(&input_device_ids),
            };
            if let Some(directory) = directory {
                return match continuity::save(directory, &evidence) {
                    Ok(file) => Err(format!("{error}; acquisition evidence: {file}")),
                    Err(save_error) => Err(format!("{error}; {save_error}")),
                };
            }
            return Err(continuity::report_error(None, &evidence));
        }
    };
    Ok(NativeCapture {
        input_device_ids,
        output_device_id,
        rate: request.sample_rate_hz,
        storage: captured,
        input_sample_formats,
        output_sample_format,
    })
}

#[cfg(test)]
#[path = "multi_capture_tests.rs"]
mod tests;
