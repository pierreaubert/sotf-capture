//! Pre-capture device wake-up for sleeping USB chains.
//!
//! USB microphones and interfaces sometimes sleep through the first stream
//! open: the take records digital silence (or a partial burst) and fails
//! the lag gate after seconds of wasted capture. Before the take loop,
//! [`wake_audio_chain`] plays a short noise burst and records it, retrying
//! until lag lock proves the chain is streaming, inside a bounded time
//! budget. Exhaustion only warns — the take gates remain the hard gate and
//! refuse a truly dead chain with the actionable error.

// Rust guideline compliant 2026-02-21

use super::consts::CANCELLED_ERR;
use super::playback::SweepPlayback;
use super::quality::{check_lag_lock, estimate_lag_or_advise};
use super::recorder::SweepRecorder;
use super::types::{CancelFlag, cancel_requested, cancel_wait};
use super::write::write_temp_wav;
use std::time::{Duration, Instant};

/// Wake-up burst length in seconds: long enough for a confident lag lock,
/// short enough that several attempts fit the budget.
pub(crate) const WAKE_BURST_SECS: f32 = 0.25;

/// Total wake-up time budget in seconds. USB sleep clears within a couple
/// of attempts; beyond this the chain is broken, not sleepy.
pub(crate) const WAKE_BUDGET_SECS: f64 = 5.0;

/// Upper ceiling for wake bursts; the configured capture ceiling may be lower.
const WAKE_PEAK: f32 = 0.5;

/// Deterministic burst seed: every wake-up plays the identical burst.
const WAKE_SEED: u64 = 0x9A4E_0001;

/// Fade length at each burst end, suppressing onset/offset clicks.
const WAKE_FADE_MS: f32 = 5.0;

/// Extra record time past the burst in each attempt, covering output
/// latency and room tail.
pub(crate) const WAKE_TAIL_SECS: f64 = 0.5;

/// Settle sleeps around each attempt (shorter than take captures: the
/// burst is short and local USB latency is milliseconds).
const WAKE_PRE_SETTLE_MS: u64 = 50;
const WAKE_POST_SETTLE_MS: u64 = 150;

/// Backoff between attempts when a stream fails to start or stop: a
/// sleeping device may refuse the first open.
pub(crate) const WAKE_ERROR_BACKOFF_MS: u64 = 200;

/// Deterministic wake-up burst: white noise with raised-cosine fades,
/// peak-normalized to the capture ceiling (at most [`WAKE_PEAK`]). White correlates unambiguously for
/// the lag lock; pink would waste energy where short windows resolve
/// poorly.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "The fixed positive fade rounds to bounded whole frames; taper phases use f32 audio arithmetic."
)]
pub(crate) fn wake_burst_samples(sample_rate: u32, amplitude: f32) -> Vec<f32> {
    if !amplitude.is_finite() || amplitude <= 0.0 {
        return Vec::new();
    }
    let mut burst = math_audio_dsp::signals::gen_white_noise_seeded(
        1.0,
        sample_rate,
        WAKE_BURST_SECS,
        WAKE_SEED,
    );
    let peak = burst
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    if peak > 0.0 {
        let gain = amplitude.min(WAKE_PEAK) / peak;
        for sample in &mut burst {
            *sample *= gain;
        }
    }
    let len = burst.len();
    let fade = ((WAKE_FADE_MS / 1000.0 * sample_rate as f32).round() as usize)
        .min(len / 2)
        .max(1);
    for (index, sample) in burst.iter_mut().enumerate() {
        let edge = index.min(len - 1 - index).min(fade);
        if edge < fade {
            let phase = std::f32::consts::PI * edge as f32 / fade as f32;
            *sample *= 0.5 - 0.5 * phase.cos();
        }
    }
    burst
}

/// Play/record short bursts until lag lock proves the chain is awake.
///
/// Runs at most `budget` of attempts, then returns `Ok` either way: a
/// sleeping chain usually answers within one or two bursts, while a truly
/// dead chain is refused later by the take gates with the actionable
/// error. Returns `Err` only when `cancel` is set.
#[allow(clippy::too_many_arguments)]
pub(super) fn wake_audio_chain(
    playback: &mut impl SweepPlayback,
    recorder: &mut impl SweepRecorder,
    sample_rate: u32,
    amplitude: f32,
    output_channel: u16,
    input_channel: u16,
    output_device_name: Option<&str>,
    input_device_name: Option<&str>,
    cancel: Option<&CancelFlag>,
    budget: Duration,
    log_tag: &str,
) -> Result<(), String> {
    let burst = wake_burst_samples(sample_rate, amplitude);
    if burst.is_empty() {
        return Ok(());
    }
    let burst_wav = write_temp_wav(&burst, sample_rate, 1)?;
    let record_secs = f64::from(WAKE_BURST_SECS) + WAKE_TAIL_SECS;
    let deadline = Instant::now() + budget;
    let mut attempts = 0_u32;
    while Instant::now() < deadline {
        attempts += 1;
        if cancel_requested(cancel) {
            return Err(CANCELLED_ERR.to_string());
        }
        let awake = wake_attempt(
            playback,
            recorder,
            burst_wav.path(),
            &burst,
            sample_rate,
            record_secs,
            output_channel,
            input_channel,
            output_device_name,
            input_device_name,
            cancel,
            log_tag,
        );
        if cancel_requested(cancel) {
            return Err(CANCELLED_ERR.into());
        }
        if awake {
            if attempts > 1 {
                log::info!("[{log_tag}] Audio chain awake after {attempts} wake-up attempts");
            } else {
                log::debug!("[{log_tag}] Audio chain awake on first wake-up attempt");
            }
            return Ok(());
        }
    }
    log::warn!(
        "[{log_tag}] Audio chain did not respond within {:.0}s of wake-up probes — proceeding; the take gates will refuse a dead chain",
        budget.as_secs_f64(),
    );
    Ok(())
}

/// One play/record/lock attempt. Returns true on lag lock; every failure
/// mode (stream errors, silence, weak lock) returns false for retry.
#[allow(clippy::too_many_arguments)]
fn wake_attempt(
    playback: &mut impl SweepPlayback,
    recorder: &mut impl SweepRecorder,
    burst_wav: &std::path::Path,
    burst: &[f32],
    sample_rate: u32,
    record_secs: f64,
    output_channel: u16,
    input_channel: u16,
    output_device_name: Option<&str>,
    input_device_name: Option<&str>,
    cancel: Option<&CancelFlag>,
    log_tag: &str,
) -> bool {
    if recorder
        .start(
            input_device_name,
            &[input_channel],
            sample_rate,
            record_secs,
        )
        .is_err()
    {
        let _ = cancel_wait(Duration::from_millis(WAKE_ERROR_BACKOFF_MS), cancel);
        return false;
    }
    if cancel_wait(Duration::from_millis(WAKE_PRE_SETTLE_MS), cancel).is_err() {
        let _ = recorder.stop();
        return false;
    }
    if playback
        .start(burst_wav, output_device_name, output_channel, sample_rate)
        .is_err()
    {
        let _ = recorder.stop();
        let _ = cancel_wait(Duration::from_millis(WAKE_ERROR_BACKOFF_MS), cancel);
        return false;
    }
    let timeout = Duration::from_secs_f64(f64::from(WAKE_BURST_SECS) + 2.0);
    let started = Instant::now();
    while !playback.is_finished() && started.elapsed() < timeout {
        if recorder.check_health().is_err()
            || playback.check_health().is_err()
            || cancel_wait(Duration::from_millis(20), cancel).is_err()
        {
            break;
        }
    }
    let completed = playback.is_finished();
    let healthy = recorder.check_health().is_ok() && playback.check_health().is_ok();
    let tail = cancel_wait(Duration::from_millis(WAKE_POST_SETTLE_MS), cancel);
    let playback_stop = playback.stop();
    let take = recorder.stop();
    if !completed || !healthy || tail.is_err() || playback_stop.is_err() || cancel_requested(cancel)
    {
        return false;
    }
    let Ok(take) = take else {
        return false;
    };
    let Some(input_samples) = take.recorded_per_channel.first() else {
        return false;
    };
    match estimate_lag_or_advise(burst, input_samples, log_tag) {
        Ok(lag) => check_lag_lock(&lag, log_tag).is_ok(),
        Err(_) => false,
    }
}
