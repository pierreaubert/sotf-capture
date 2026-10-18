//! Track final output presentation separately from callback submission.

// Rust guideline compliant 2026-02-21

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub(super) struct PresentationClock {
    origin: Instant,
    deadline_ns: AtomicU64,
}

fn completion_time(
    now: Duration,
    latency: Option<Duration>,
    frames: usize,
    rate: u32,
) -> Option<Duration> {
    if rate == 0 {
        return None;
    }
    now.checked_add(latency?)?
        .checked_add(Duration::from_secs_f64(
            f64::from(u32::try_from(frames).ok()?) / f64::from(rate),
        ))
}

impl PresentationClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
            deadline_ns: AtomicU64::new(0),
        }
    }

    pub fn submit(
        &self,
        info: &cpal::OutputCallbackInfo,
        frames: usize,
        rate: u32,
        failed: &AtomicBool,
    ) {
        let timestamp = info.timestamp();
        let latency = timestamp.playback.duration_since(&timestamp.callback);
        let deadline = completion_time(self.origin.elapsed(), latency, frames, rate)
            .and_then(|time| u64::try_from(time.as_nanos()).ok());
        if let Some(deadline) = deadline {
            // Only the callback containing the final stimulus frame publishes
            // this deadline. Later silence callbacks must not postpone it.
            self.deadline_ns.store(deadline.max(1), Ordering::Release);
        } else {
            failed.store(true, Ordering::Release);
        }
    }

    pub fn is_finished(&self) -> bool {
        let deadline = self.deadline_ns.load(Ordering::Acquire);
        deadline != 0 && self.origin.elapsed().as_nanos() >= u128::from(deadline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_accounts_for_queue_latency_and_final_frame_position() {
        assert_eq!(
            completion_time(
                Duration::from_secs(2),
                Some(Duration::from_millis(750)),
                480,
                48000
            ),
            Some(Duration::from_millis(2760))
        );
        assert_eq!(completion_time(Duration::ZERO, None, 480, 48000), None);
        assert_eq!(
            completion_time(Duration::ZERO, Some(Duration::ZERO), 480, 0),
            None
        );
        assert!(!PresentationClock::new().is_finished());
    }
}
