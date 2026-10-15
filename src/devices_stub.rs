//! Audio device facade for builds without hardware I/O.
//!
//! Provides the same data types as the cpal-backed `devices` module
//! ([`AudioDevice`], [`AudioConfig`], [`AudioState`], [`SharedAudioState`])
//! without depending on cpal. Active when the `capture-io` feature is off,
//! i.e. on tvOS, where cpal has no backend. tvOS exposes no microphone to
//! third-party apps, so this facade lists the system output route only and
//! refuses every input operation with an actionable message; device lookup
//! helpers shared with the emulated backends keep their signatures.

// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Represents information about an audio device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDevice {
    pub device_id: Option<String>,
    pub name: String,
    pub display_info: Option<String>,
    pub is_input: bool,
    pub is_default: bool,
    pub supported_configs: Vec<AudioConfig>,
    pub default_config: Option<AudioConfig>,
    pub available_sample_rates: Vec<u32>,
}

/// Represents audio configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub buffer_size: Option<u32>,
    pub sample_format: String,
}

/// State for storing the currently selected audio configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioState {
    pub selected_input_device: Option<String>,
    pub selected_output_device: Option<String>,
    pub input_config: Option<AudioConfig>,
    pub output_config: Option<AudioConfig>,
}

pub type SharedAudioState = Arc<Mutex<AudioState>>;

/// Stable identifier of the single tvOS system output route.
pub const TVOS_SYSTEM_OUTPUT_ID: &str = "tvos-system-output";
/// Display name of the single tvOS system output route.
pub const TVOS_SYSTEM_OUTPUT_NAME: &str = "System Output";
/// Sample rates a tvOS HDMI route runs without OS resampling.
const TVOS_SAMPLE_RATES: &[u32] = &[44_100, 48_000];

pub const ASIO_DEVICE_PREFIX: &str = "ASIO:";

/// Whether an identifier names a Windows ASIO device (never on tvOS).
#[must_use]
pub fn is_asio_device(identifier: &str) -> bool {
    identifier.len() >= ASIO_DEVICE_PREFIX.len()
        && identifier[..ASIO_DEVICE_PREFIX.len()].eq_ignore_ascii_case(ASIO_DEVICE_PREFIX)
}

/// Strip a Windows ASIO device prefix, if present.
#[must_use]
pub fn strip_asio_prefix(identifier: &str) -> &str {
    if is_asio_device(identifier) {
        &identifier[ASIO_DEVICE_PREFIX.len()..]
    } else {
        identifier
    }
}

/// List ASIO devices (always empty without hardware I/O).
#[must_use]
pub fn list_asio_devices() -> Vec<String> {
    Vec::new()
}

fn tvos_default_output_config() -> AudioConfig {
    AudioConfig {
        sample_rate: 48_000,
        channels: 2,
        buffer_size: None,
        sample_format: "f32".to_string(),
    }
}

fn matches_tvos_output(identifier: &str) -> bool {
    identifier == TVOS_SYSTEM_OUTPUT_ID || identifier.eq_ignore_ascii_case(TVOS_SYSTEM_OUTPUT_NAME)
}

fn tvos_config_supported(config: &AudioConfig) -> bool {
    TVOS_SAMPLE_RATES.contains(&config.sample_rate)
        && (1..=8).contains(&config.channels)
        && config.sample_format == "f32"
}

/// tvOS exposes the system HDMI output route as a single selectable
/// device; the input list is always empty (no third-party mic access).
///
/// # Errors
///
/// Never fails; the `Result` mirrors the fallible cpal-backed signature.
pub fn get_audio_devices() -> Result<HashMap<String, Vec<AudioDevice>>, String> {
    let mut devices_map = HashMap::new();
    devices_map.insert("input".to_string(), Vec::new());
    devices_map.insert(
        "output".to_string(),
        vec![AudioDevice {
            device_id: Some(TVOS_SYSTEM_OUTPUT_ID.to_string()),
            name: TVOS_SYSTEM_OUTPUT_NAME.to_string(),
            display_info: Some("tvOS Audio".to_string()),
            is_input: false,
            is_default: true,
            supported_configs: vec![tvos_default_output_config()],
            default_config: Some(tvos_default_output_config()),
            available_sample_rates: TVOS_SAMPLE_RATES.to_vec(),
        }],
    );
    Ok(devices_map)
}

#[must_use]
pub fn get_device_supported_sample_rates(device_identifier: Option<&str>) -> Option<Vec<u32>> {
    if device_identifier.is_none_or(matches_tvos_output) {
        Some(TVOS_SAMPLE_RATES.to_vec())
    } else {
        None
    }
}

#[must_use]
pub fn get_device_current_sample_rate(device_identifier: Option<&str>) -> Option<u32> {
    if device_identifier.is_none_or(matches_tvos_output) {
        Some(48_000)
    } else {
        None
    }
}

#[must_use]
pub fn verify_working_sample_rate(
    device_identifier: Option<&str>,
    requested_rate: u32,
    requested_channels: usize,
) -> Option<u32> {
    if device_identifier.is_some_and(|identifier| !matches_tvos_output(identifier)) {
        return None;
    }
    if requested_channels == 0 || requested_channels > 8 {
        return None;
    }
    if TVOS_SAMPLE_RATES.contains(&requested_rate) {
        Some(requested_rate)
    } else {
        Some(48_000)
    }
}

#[must_use]
pub fn is_null_device(_name: &str) -> bool {
    false
}

/// Helpful not-found message listing available devices (shared with fakes).
#[cfg(any(test, feature = "test-util"))]
pub(crate) fn device_not_found_message(
    identifier: &str,
    is_input: bool,
    device_info: &[(String, String)],
) -> String {
    let device_type = if is_input { "input" } else { "output" };
    let mut names: Vec<String> = device_info
        .iter()
        .filter_map(|(_, name)| {
            let trimmed = name.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        })
        .collect();
    names.sort();
    names.dedup();
    format!(
        "Audio device '{}' not found. Available {} devices ({} total): {}",
        identifier,
        device_type,
        device_info.len(),
        names.join(", "),
    )
}

/// Select the tvOS output route and store its configuration.
///
/// # Errors
///
/// Returns a message for input selection (unsupported on tvOS), unknown
/// devices, unsupported configurations, or a poisoned state lock.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Owned device/config mirror the cpal-backed signature so callers stay portable."
)]
pub fn set_audio_device(
    device_identifier: String,
    is_input: bool,
    config: AudioConfig,
    audio_state: &SharedAudioState,
) -> Result<String, String> {
    if is_input {
        return Err(
            "tvOS exposes no microphone input to third-party apps; capture is unavailable"
                .to_string(),
        );
    }
    if !matches_tvos_output(&device_identifier) {
        return Err(format!("Device '{device_identifier}' not found"));
    }
    if !tvos_config_supported(&config) {
        return Err(format!(
            "Configuration not supported by tvOS output '{}': sample_rate={}, channels={}, format={}",
            device_identifier, config.sample_rate, config.channels, config.sample_format
        ));
    }

    let mut state = audio_state
        .lock()
        .map_err(|e| format!("Failed to lock audio state: {e}"))?;
    state.selected_output_device = Some(TVOS_SYSTEM_OUTPUT_ID.to_string());
    state.output_config = Some(config.clone());

    Ok(format!(
        "Successfully configured output device '{}' with sample_rate={}, channels={}, format={}",
        device_identifier, config.sample_rate, config.channels, config.sample_format
    ))
}

/// Read the stored audio configuration.
///
/// # Errors
///
/// Returns a message when the state lock is poisoned.
pub fn get_audio_config(audio_state: &SharedAudioState) -> Result<AudioState, String> {
    let state = audio_state
        .lock()
        .map_err(|e| format!("Failed to lock audio state: {e}"))?;
    Ok(state.clone())
}

/// Describe the tvOS output route as JSON.
///
/// # Errors
///
/// Returns a message for input queries (unsupported on tvOS) and unknown
/// devices.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Owned identifier mirrors the cpal-backed signature so callers stay portable."
)]
pub fn get_device_properties(
    device_identifier: String,
    is_input: bool,
) -> Result<serde_json::Value, String> {
    if is_input {
        return Err(
            "tvOS exposes no microphone input to third-party apps; capture is unavailable"
                .to_string(),
        );
    }
    if !matches_tvos_output(&device_identifier) {
        return Err(format!("Device '{device_identifier}' not found"));
    }

    Ok(serde_json::json!({
        "id": TVOS_SYSTEM_OUTPUT_ID,
        "name": TVOS_SYSTEM_OUTPUT_NAME,
        "type": "output",
        "sample_rates": TVOS_SAMPLE_RATES,
        "channels": 8,
        "sample_format": "f32",
    }))
}
