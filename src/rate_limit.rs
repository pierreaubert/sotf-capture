// Rate-limited logging helpers for audio-thread error paths.
//
// Mirrors sotf-host's lightweight pattern: a static AtomicU64 stores the next
// allowed timestamp for each call site. The check itself is lock-free and
// allocation-free, so it is suitable for callback and polling paths.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[inline]
fn now_ns() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    u64::try_from(Instant::now().saturating_duration_since(*start).as_nanos()).unwrap_or(u64::MAX)
}

#[inline]
pub fn allow(gate: &AtomicU64, interval_ns: u64) -> bool {
    let now = now_ns();
    let prev = gate.load(Ordering::Relaxed);
    if now < prev {
        return false;
    }

    gate.compare_exchange(
        prev,
        now.saturating_add(interval_ns),
        Ordering::Relaxed,
        Ordering::Relaxed,
    )
    .is_ok()
}

#[macro_export]
macro_rules! rate_limited_log {
    ($level:ident, $secs:expr, $($arg:tt)+) => {{
        static GATE: ::std::sync::atomic::AtomicU64 =
            ::std::sync::atomic::AtomicU64::new(0);
        #[allow(
            trivial_numeric_casts,
            reason = "Macro callers may supply seconds using different numeric types."
        )]
        let interval_secs = $secs as u64;
        if $crate::rate_limit::allow(
            &GATE,
            interval_secs.saturating_mul(1_000_000_000),
        ) {
            ::log::$level!($($arg)+);
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::allow;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn gate_allows_once_per_interval() {
        let gate = AtomicU64::new(0);

        assert!(allow(&gate, 1_000_000_000));
        assert!(!allow(&gate, 1_000_000_000));
    }
}
