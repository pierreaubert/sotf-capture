//! Shared acquisition resource limits, checked before signal generation.

// Rust guideline compliant 2026-02-21

pub(crate) const MAX_CAPTURE_FRAMES: usize = 16_000_000;

pub(crate) fn sequential_frames(
    signal: usize,
    silence: usize,
    outputs: usize,
) -> Result<usize, String> {
    signal
        .checked_add(silence)
        .and_then(|segment| segment.checked_mul(outputs))
        .and_then(|sequence| sequence.checked_add(silence))
        .ok_or_else(|| "auxiliary frame count overflow".into())
}

// Budget simultaneous acquisition storage, including rings and the caller's
// reference plus the playback-owned stimulus. Analysis has a separate lifetime.
const MAX_CAPTURE_BYTES: usize = 512 * 1024 * 1024;

// A float batch is one 4 KiB page; each buffered file write amortizes two batches.
pub(crate) const FILE_CHUNK_FRAMES: usize = 1024;
pub(crate) const FILE_WRITER_BUFFER_BYTES: usize = 8 * 1024;
// Small captures retain at least 256 KiB of queue payload; larger captures use
// the rate/input-scaled one-second capacity below.
pub(crate) const FILE_QUEUE_CHUNKS: usize = 64;

/// Match the input rings' one-second burst tolerance, with a small-case floor.
pub(crate) fn file_queue_chunks(inputs: usize, rate: u32) -> Result<usize, String> {
    inputs
        .checked_mul((rate as usize).div_ceil(FILE_CHUNK_FRAMES))
        .map(|chunks| chunks.max(FILE_QUEUE_CHUNKS))
        .ok_or_else(|| "capture writer queue size overflow".into())
}

/// Bound live streaming payloads, independently of the total recording length.
pub(crate) fn file_capture_bytes(
    inputs: usize,
    rate: u32,
    stimulus: usize,
) -> Result<usize, String> {
    let queue_chunks = file_queue_chunks(inputs, rate)?;
    let samples = inputs
        .checked_mul(rate as usize)
        .and_then(|rings| {
            stimulus
                .checked_mul(2)
                .and_then(|signal| rings.checked_add(signal))
        })
        .and_then(|samples| {
            inputs
                .checked_add(queue_chunks.checked_add(2)?)
                .and_then(|chunks| chunks.checked_mul(FILE_CHUNK_FRAMES))
                .and_then(|chunks| samples.checked_add(chunks))
        })
        .ok_or("file capture storage size overflow")?;
    // Each mono WAV has an 8 KiB buffered writer. Reserve another MiB for
    // channels, counters, and headers, beyond the explicit f32 payloads.
    let bytes = samples
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| {
            inputs
                .checked_mul(FILE_WRITER_BUFFER_BYTES)
                .and_then(|writers| bytes.checked_add(writers))
        })
        .and_then(|bytes| bytes.checked_add(1024 * 1024))
        .ok_or("file capture storage size overflow")?;
    if bytes > MAX_CAPTURE_BYTES {
        return Err("file capture exceeds the 512 MiB live acquisition budget".into());
    }
    Ok(bytes)
}

#[cfg(any(feature = "capture-io", test))]
pub(crate) fn capture_bytes(
    inputs: usize,
    frames: usize,
    rate: u32,
    stimulus: usize,
) -> Result<usize, String> {
    let bytes = frames
        .checked_add(rate as usize)
        .and_then(|per_input| per_input.checked_mul(inputs))
        .and_then(|samples| stimulus.checked_mul(2).and_then(|s| samples.checked_add(s)))
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>()))
        .ok_or("capture storage size overflow")?;
    if bytes > MAX_CAPTURE_BYTES {
        return Err("capture exceeds the 512 MiB total acquisition budget; reduce duration, rate, or input count".into());
    }
    Ok(bytes)
}

/// Budget a sequential auxiliary pass, including input startup and drain.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "Positive finite frame counts are bounded to 16 million before conversion; route count is at most 64."
)]
pub(crate) fn auxiliary_frames(
    rate: u32,
    outputs: usize,
    signal_seconds: f64,
    silence_seconds: f64,
) -> Result<usize, String> {
    if !(8000..=384_000).contains(&rate)
        || !(1..=64).contains(&outputs)
        || !signal_seconds.is_finite()
        || signal_seconds <= 0.0
        || !silence_seconds.is_finite()
        || silence_seconds < 0.0
    {
        return Err("invalid auxiliary rate, routes, or duration".into());
    }
    let rate = f64::from(rate);
    let signal = signal_seconds * rate;
    let silence = silence_seconds * rate;
    // Five seconds for startup and one for drain reserve capture storage
    // before opening streams. Reject floats before any saturating integer cast.
    let total = silence + outputs as f64 * (signal + silence) + 6.0 * rate;
    if signal < 2.0 || total > MAX_CAPTURE_FRAMES as f64 || !total.is_finite() {
        return Err("auxiliary pass exceeds bounded capture storage at this sample rate".into());
    }
    sequential_frames(signal as usize, silence as usize, outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_extreme_nonfinite_and_aggregate_durations() {
        for duration in [1e20, f64::INFINITY, f64::NAN, 0.0, -1.0] {
            assert!(auxiliary_frames(48000, 2, duration, 0.1).is_err());
        }
        assert!(auxiliary_frames(48000, 64, 10.0, 0.5).is_err());
        assert!(auxiliary_frames(48000, 2, 0.1, 1e20).is_err());
        assert_eq!(auxiliary_frames(48000, 64, 2.0, 0.5).unwrap(), 7_704_000);
    }

    #[test]
    fn total_budget_counts_every_input_ring_and_checked_product() {
        assert!(capture_bytes(33, 16_000_000, 384_000, 1000).is_err());
        assert!(capture_bytes(usize::MAX, 16_000_000, 384_000, 1000).is_err());
        assert_eq!(capture_bytes(4, 96000, 48000, 48000).unwrap(), 2_688_000);
    }

    #[test]
    fn file_budget_supports_large_takes_with_bounded_live_storage() {
        assert!(capture_bytes(33, 4_000_000, 48000, 3_520_000).is_err());
        assert!(file_capture_bytes(33, 48000, 3_520_000).unwrap() < 64 * 1024 * 1024);
        assert!(
            file_capture_bytes(33, 384_000, MAX_CAPTURE_FRAMES - 384_000 * 10).unwrap()
                < MAX_CAPTURE_BYTES
        );
        assert!(file_capture_bytes(usize::MAX, 48000, 1024).is_err());
        assert!(file_capture_bytes(33, 48000, usize::MAX).is_err());
    }
}
