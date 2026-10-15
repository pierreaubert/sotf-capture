//! iOS audio-route helpers for measurement capture.
//!
//! iOS exposes a single audio route (built-in mic/speaker, headset, or one
//! USB interface) instead of the desktop device list, and the OS owns route
//! selection through `AVAudioSession`. This module encodes that model: the
//! sample rates iOS actually runs, the channel limits of the built-in path,
//! and the app-side session setup capture depends on. The Swift bootstrap
//! (session category, microphone permission, `Info.plist` key) is in
//! README.md §iOS; call it before opening any cpal stream.
//!
//! All items are pure and platform-independent so hosts can unit-test the
//! negotiation logic; they describe OS behavior, not the current route.

// Rust guideline compliant 2026-02-21

/// Sample rates the iOS audio route runs without OS resampling.
///
/// iOS hardware natively runs at 44.1 kHz or 48 kHz depending on the route;
/// requesting anything else makes the OS resample behind cpal, which
/// corrupts sweep timing. Prefer the active route rate (see `docs/ios.md`);
/// use [`DEFAULT_SAMPLE_RATE_HZ`] when the route rate is unknown.
pub const IOS_SAMPLE_RATES_HZ: &[u32] = &[44_100, 48_000];

/// Fallback rate when the active route rate is unknown.
///
/// 48 kHz matches the measurement sweep grid used across SOTF tooling and
/// is the rate most iOS routes (including USB interfaces) prefer.
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 48_000;

/// Maximum input channels on the built-in/headset path.
///
/// The built-in microphone is mono; stereo recording and anything wider
/// needs a USB audio interface via the Camera Adapter. The built-in
/// speaker/headset output path is stereo.
pub const BUILTIN_INPUT_CHANNELS: u16 = 1;

/// Negotiate the capture rate against what iOS routes support.
///
/// Returns `requested` when it is a natively supported route rate;
/// otherwise returns [`DEFAULT_SAMPLE_RATE_HZ`] so the sweep grid stays
/// exact instead of drifting through hidden OS resampling. A `None`
/// request (route default) also resolves to the default rate.
#[must_use]
pub fn negotiate_sample_rate(requested: Option<u32>) -> u32 {
    match requested {
        Some(rate) if IOS_SAMPLE_RATES_HZ.contains(&rate) => rate,
        _ => DEFAULT_SAMPLE_RATE_HZ,
    }
}

/// Validate an input channel count for the built-in/headset route.
///
/// # Errors
///
/// Returns a message when `channels` is zero or exceeds
/// [`BUILTIN_INPUT_CHANNELS`]. USB interfaces negotiate their own width
/// through cpal; this guard covers only the built-in path, where asking
/// for stereo silently duplicates or drops the mono capsule feed.
pub fn check_builtin_input_channels(channels: u16) -> Result<(), String> {
    if channels == 0 || channels > BUILTIN_INPUT_CHANNELS {
        return Err(format!(
            "Built-in iOS input supports {BUILTIN_INPUT_CHANNELS} channel(s), got {channels}; \
             use a USB audio interface for multichannel capture"
        ));
    }
    Ok(())
}

/// App-side session setup capture requires on iOS.
///
/// cpal opens the `RemoteIO` unit but cannot set the session category, mode,
/// or microphone permission; the hosting app must apply this checklist
/// (Swift snippet in README.md §iOS) before the first stream opens,
/// otherwise input streams fail with permission errors.
pub const SESSION_SETUP_CHECKLIST: &str = "AVAudioSession: category .playAndRecord, mode .measurement, \
     setPreferredSampleRate, requestRecordPermission, NSMicrophoneUsageDescription in Info.plist";

#[cfg(test)]
mod tests {
    use super::{check_builtin_input_channels, negotiate_sample_rate};

    #[test]
    fn keeps_natively_supported_rates() {
        assert_eq!(negotiate_sample_rate(Some(44_100)), 44_100);
        assert_eq!(negotiate_sample_rate(Some(48_000)), 48_000);
    }

    #[test]
    fn falls_back_to_default_rate() {
        assert_eq!(
            negotiate_sample_rate(Some(96_000)),
            super::DEFAULT_SAMPLE_RATE_HZ
        );
        assert_eq!(negotiate_sample_rate(None), super::DEFAULT_SAMPLE_RATE_HZ);
    }

    #[test]
    fn validates_builtin_input_width() {
        assert!(check_builtin_input_channels(1).is_ok());
        assert!(check_builtin_input_channels(0).is_err());
        assert!(check_builtin_input_channels(2).is_err());
    }
}
