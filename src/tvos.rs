//! tvOS audio-route helpers for playback-only builds.
//!
//! tvOS exposes no microphone to third-party apps, so `sotf-capture` builds
//! there without the `capture-io` feature: signal generation, analysis, and
//! session planning compile, while device I/O, playback, and recording do
//! not. This module encodes the output side of that model: the sample rates
//! a tvOS HDMI route runs and the channel width multichannel LPCM output
//! supports. There is no input side to negotiate.

// Rust guideline compliant 2026-02-21

/// Sample rates a tvOS HDMI route runs without OS resampling.
///
/// tvOS routes natively run at 44.1 kHz or 48 kHz; requesting anything
/// else makes the OS resample behind the stream. Prefer 48 kHz, the rate
/// most tvOS routes (and all SOTF measurement grids) use.
pub const TVOS_SAMPLE_RATES_HZ: &[u32] = &[44_100, 48_000];

/// Fallback rate when the active route rate is unknown.
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 48_000;

/// Maximum output channels on the tvOS HDMI route.
///
/// tvOS outputs multichannel LPCM over HDMI, so surround layouts fit;
/// recording channel counts do not apply — there is no input.
pub const HDMI_OUTPUT_CHANNELS: u16 = 8;

/// Negotiate a playback rate against what tvOS routes support.
///
/// Returns `requested` when it is a natively supported route rate;
/// otherwise returns [`DEFAULT_SAMPLE_RATE_HZ`] so output stays exact
/// instead of drifting through hidden OS resampling. A `None` request
/// (route default) also resolves to the default rate.
#[must_use]
pub fn negotiate_sample_rate(requested: Option<u32>) -> u32 {
    match requested {
        Some(rate) if TVOS_SAMPLE_RATES_HZ.contains(&rate) => rate,
        _ => DEFAULT_SAMPLE_RATE_HZ,
    }
}

/// Validate an output channel count for the tvOS HDMI route.
///
/// # Errors
///
/// Returns a message when `channels` is zero or exceeds
/// [`HDMI_OUTPUT_CHANNELS`].
pub fn check_hdmi_output_channels(channels: u16) -> Result<(), String> {
    if channels == 0 || channels > HDMI_OUTPUT_CHANNELS {
        return Err(format!(
            "tvOS HDMI output supports 1-{HDMI_OUTPUT_CHANNELS} channels, got {channels}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_hdmi_output_channels, negotiate_sample_rate};

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
    fn validates_hdmi_output_width() {
        assert!(check_hdmi_output_channels(2).is_ok());
        assert!(check_hdmi_output_channels(8).is_ok());
        assert!(check_hdmi_output_channels(0).is_err());
        assert!(check_hdmi_output_channels(9).is_err());
    }
}
