//! Simultaneous, independently clocked measurement input streams.
//!
//! This blocking control-thread API captures raw samples only. Opening every
//! input before playback prevents sequential recordings from being mistaken for
//! simultaneous takes. Acoustic chirps, clock correction and uncertainty belong
//! to the caller's measurement protocol; equal nominal rates do not imply sync.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::CancelFlag;
use super::measurement::measurement_sample_format_rank;

// Four microphones is the capture protocol limit. Grouped input frames fit in
// a fixed array so callbacks never allocate, even for aggregate devices.
const MAX_MICS: usize = 4;
// Bound configuration-driven memory (at most 256 MiB for planar capture).
const MAX_FRAMES: usize = 16_000_000;
const START_TIMEOUT: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(5);
// Keep recording after the final output buffer is submitted to the device.
const DRAIN_TIME: Duration = Duration::from_millis(500);

/// A microphone's exact device selector and zero-based input channel.
#[derive(Debug, Clone)]
pub struct CaptureInput {
    /// Exact device ID or unique enumerated name; ambiguous names are rejected.
    pub device: String,
    /// Physical device input channel.
    pub channel: u16,
}

/// A bounded region of the mono stimulus sent to a different output channel.
#[derive(Debug, Clone)]
pub struct CaptureOutputSegment {
    /// Inclusive stimulus frame index.
    pub start_frame: usize,
    /// Exclusive stimulus frame index.
    pub end_frame: usize,
    /// Physical output channel for this region.
    pub channel: u16,
}

/// One source stimulus recorded simultaneously on every requested input.
#[derive(Debug)]
pub struct MultiCaptureRequest {
    /// Two to four input routes, ordered as the returned recordings.
    pub inputs: Vec<CaptureInput>,
    /// Exact playback device ID or unique enumerated name.
    pub output_device: String,
    /// Physical output channel; all other channels are silent.
    pub output_channel: u16,
    /// Sorted, nonoverlapping output overrides (at most 16), e.g. timing chirps.
    pub output_overrides: Vec<CaptureOutputSegment>,
    /// Required nominal rate on every stream; there is no rate fallback.
    pub sample_rate_hz: u32,
    /// Mono stimulus, including caller-defined timing chirps and silence.
    pub stimulus: Vec<f32>,
}

/// Raw device-clock samples, before acoustic timing correction.
#[derive(Debug)]
pub struct MultiCaptureResult {
    /// Actual device IDs in request order, independent of user-facing names.
    pub input_device_ids: Vec<String>,
    /// Actual playback device ID.
    pub output_device_id: String,
    /// Nominal rate confirmed for all configured streams.
    pub sample_rate_hz: u32,
    /// Planar samples in request order; independent clocks can differ in length.
    pub recordings: Vec<Vec<f32>>,
    /// Negotiated sample formats in request order, for recording provenance.
    pub input_sample_formats: Vec<String>,
    /// Negotiated output sample format.
    pub output_sample_format: String,
}

#[derive(Debug)]
struct InputGroup {
    name: String,
    routes: Vec<(usize, u16)>,
}

fn group_inputs(request: &MultiCaptureRequest) -> Result<Vec<InputGroup>, String> {
    if !(2..=MAX_MICS).contains(&request.inputs.len()) {
        return Err("simultaneous capture requires two to four inputs".into());
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

fn exact_device(
    devices: impl Iterator<Item = cpal::Device>,
    name: &str,
) -> Result<cpal::Device, String> {
    let mut devices: Vec<_> = devices.collect();
    if let Some(index) = devices
        .iter()
        .position(|device| device.id().is_ok_and(|id| id.to_string() == name))
    {
        return Ok(devices.remove(index));
    }
    let mut found = None;
    for device in devices {
        if device
            .description()
            .map_err(|e| format!("cannot identify audio device: {e}"))?
            .name()
            == name
        {
            if found.is_some() {
                return Err(format!(
                    "audio device name is ambiguous: {name}; select its stable device ID"
                ));
            }
            found = Some(device);
        }
    }
    found.ok_or_else(|| format!("audio device not found: {name}"))
}

fn exact_config(
    supported: impl Iterator<Item = cpal::SupportedStreamConfigRange>,
    required_channels: u16,
    rate: u32,
) -> Result<cpal::SupportedStreamConfig, String> {
    supported
        .filter(|c| c.channels() >= required_channels && c.channels() <= 64
            && c.min_sample_rate() <= rate && c.max_sample_rate() >= rate
            && measurement_sample_format_rank(c.sample_format()) < 12)
        .min_by_key(|c| (measurement_sample_format_rank(c.sample_format()), c.channels()))
        .map(|c| c.with_sample_rate(rate))
        .ok_or_else(|| format!("device cannot capture/play {required_channels} channels at exactly {rate} Hz in a supported PCM format"))
}

// Shared pure callback cores are also exercised without opening devices.
fn capture_frames<T>(
    data: &[T],
    channels: usize,
    selected: &[u16],
    producer: &mut rtrb::Producer<[f32; MAX_MICS]>,
    failed: &AtomicBool,
) where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    if channels == 0
        || !data.len().is_multiple_of(channels)
        || selected.len() > MAX_MICS
        || selected
            .iter()
            .any(|channel| usize::from(*channel) >= channels)
    {
        failed.store(true, Ordering::Relaxed);
        return;
    }
    for frame in data.chunks_exact(channels) {
        let mut selected_frame = [0.0; MAX_MICS];
        for (slot, channel) in selected.iter().enumerate() {
            let sample = <f32 as cpal::FromSample<T>>::from_sample_(frame[usize::from(*channel)]);
            if !sample.is_finite() {
                failed.store(true, Ordering::Relaxed);
                return;
            }
            selected_frame[slot] = sample;
        }
        if producer.push(selected_frame).is_err() {
            failed.store(true, Ordering::Relaxed);
            return;
        }
    }
}

fn playback_frames<T>(
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
    mut producer: rtrb::Producer<[f32; MAX_MICS]>,
    failed: Arc<AtomicBool>,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = usize::from(config.channels);
    let data_failed = Arc::clone(&failed);
    device
        .build_input_stream::<T, _, _>(
            config,
            move |data, _| capture_frames(data, channels, &selected, &mut producer, &data_failed),
            move |_| {
                failed.store(true, Ordering::Relaxed);
            },
            None,
        )
        .map_err(|e| format!("cannot build capture input: {e}"))
}

fn output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    selected: usize,
    overrides: Vec<CaptureOutputSegment>,
    stimulus: Arc<Vec<f32>>,
    cursor: Arc<AtomicUsize>,
    failed: Arc<AtomicBool>,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let channels = usize::from(config.channels);
    let data_failed = Arc::clone(&failed);
    device
        .build_output_stream::<T, _, _>(
            config,
            move |data, _| {
                playback_frames(
                    data,
                    channels,
                    selected,
                    &overrides,
                    &stimulus,
                    &cursor,
                    &data_failed,
                )
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

struct RunningInput {
    stream: cpal::Stream,
    consumer: rtrb::Consumer<[f32; MAX_MICS]>,
    routes: Vec<(usize, u16)>,
    failed: Arc<AtomicBool>,
    name: String,
}

fn drain_inputs(
    inputs: &mut [RunningInput],
    recordings: &mut [Vec<f32>],
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
        for _ in 0..input.consumer.slots() {
            let Ok(frame) = input.consumer.pop() else {
                break;
            };
            for (slot, (index, _)) in input.routes.iter().enumerate() {
                if recordings[*index].len() >= limit {
                    return Err("capture exceeded bounded recording storage".into());
                }
                recordings[*index].push(frame[slot]);
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
    let groups = group_inputs(&request)?;
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    let host = cpal::default_host();
    let output = exact_device(
        host.output_devices().map_err(|e| e.to_string())?,
        &request.output_device,
    )?;
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
    let mut recordings = Vec::with_capacity(request.inputs.len());
    for _ in &request.inputs {
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(limit)
            .map_err(|e| format!("cannot allocate capture storage: {e}"))?;
        recordings.push(samples);
    }
    // Build every stream before any stream starts. Dropping this vector handles
    // cleanup if device setup or any later play() call fails partway through.
    for group in groups {
        let device = exact_device(
            host.input_devices().map_err(|e| e.to_string())?,
            &group.name,
        )?;
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
            input_device_ids[*index] = device_id.clone();
        }
        let (producer, consumer) = rtrb::RingBuffer::new(request.sample_rate_hz as usize);
        let failed = Arc::new(AtomicBool::new(false));
        let selected = group.routes.iter().map(|(_, channel)| *channel).collect();
        let stream = with_sample_format!(
            config.sample_format(),
            input_stream,
            &device,
            &config.config(),
            selected,
            producer,
            Arc::clone(&failed)
        )?;
        inputs.push(RunningInput {
            stream,
            consumer,
            routes: group.routes,
            failed,
            name: group.name,
        });
    }
    let stimulus = Arc::new(request.stimulus);
    let cursor = Arc::new(AtomicUsize::new(0));
    let output_failed = Arc::new(AtomicBool::new(false));
    let output_stream = with_sample_format!(
        output_config.sample_format(),
        output_stream,
        &output,
        &output_config.config(),
        usize::from(request.output_channel),
        request.output_overrides,
        Arc::clone(&stimulus),
        Arc::clone(&cursor),
        Arc::clone(&output_failed)
    )?;
    for input in &inputs {
        input
            .stream
            .play()
            .map_err(|e| format!("cannot start {}: {e}", input.name))?;
    }
    let start = Instant::now();
    // At least 250 ms of real input frames from every mic before starting output.
    while recordings
        .iter()
        .any(|samples| samples.len() < request.sample_rate_hz as usize / 4)
    {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        drain_inputs(&mut inputs, &mut recordings, limit)?;
        if start.elapsed() > START_TIMEOUT {
            return Err("capture input did not start within five seconds".into());
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    let playback_input_starts: Vec<usize> = recordings.iter().map(Vec::len).collect();
    output_stream
        .play()
        .map_err(|e| format!("cannot start playback: {e}"))?;
    let playback_start = Instant::now();
    let timeout =
        Duration::from_secs_f64(stimulus.len() as f64 / f64::from(request.sample_rate_hz))
            + START_TIMEOUT;
    let mut completed_at = None;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        if output_failed.load(Ordering::Relaxed) {
            return Err("capture playback stream failed".into());
        }
        drain_inputs(&mut inputs, &mut recordings, limit)?;
        if cursor.load(Ordering::Acquire) >= stimulus.len() {
            let completed = completed_at.get_or_insert_with(Instant::now);
            if completed.elapsed() >= DRAIN_TIME {
                break;
            }
        }
        if playback_start.elapsed() > timeout {
            return Err("capture playback stalled".into());
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    drop(output_stream);
    if output_failed.load(Ordering::Relaxed) {
        return Err("capture playback stream failed while stopping".into());
    }
    for input in &inputs {
        input
            .stream
            .pause()
            .map_err(|e| format!("cannot stop {}: {e}", input.name))?;
    }
    drain_inputs(&mut inputs, &mut recordings, limit)?;
    for (index, samples) in recordings.iter().enumerate() {
        // The 500 ms drain interval accommodates normal crystal skew. A stream
        // that stops delivering data mid-take must not be reported as successful.
        if samples.len().saturating_sub(playback_input_starts[index]) < stimulus.len() {
            return Err(format!(
                "input {} stopped before the complete stimulus was captured",
                request.inputs[index].device
            ));
        }
    }
    Ok(MultiCaptureResult {
        input_device_ids,
        output_device_id,
        sample_rate_hz: request.sample_rate_hz,
        recordings,
        input_sample_formats,
        output_sample_format,
    })
}

#[cfg(test)]
#[path = "multi_capture_tests.rs"]
mod tests;
