use super::is::is_null_device;
use super::misc::match_device_priority;
use super::misc::summarize_available_device_names;
use cpal::Device;
use cpal::traits::{DeviceTrait, HostTrait};

/// Given a cpal host, find the first non-null output device.
/// Returns the default device if it's not a null device, otherwise scans for a real one.
pub(super) fn find_real_output_device(host: &cpal::Host) -> Option<Device> {
    let default = host.default_output_device()?;
    let name = default
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_default();

    if !is_null_device(&name) {
        return Some(default);
    }

    log::warn!(
        "[AUDIO] Default output device is '{name}' (null sink), searching for a real device"
    );

    // Find the first real hardware device
    let devices = host.output_devices().ok()?;
    for dev in devices {
        let dev_name = dev
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_default();
        if !is_null_device(&dev_name) && !dev_name.is_empty() {
            log::info!("[AUDIO] Using fallback output device: '{dev_name}'");
            return Some(dev);
        }
    }

    log::warn!("[AUDIO] No real output device found, using null sink as last resort");
    Some(default)
}

/// Find a measurement device by exact ID or unique complete name.
///
/// # Errors
/// Returns an error if enumeration fails, the name is ambiguous, or no exact match exists.
pub fn find_device(
    host: &cpal::Host,
    identifier: &str,
    is_input: bool,
) -> Result<cpal::Device, String> {
    if super::is_asio_device(identifier) {
        #[cfg(all(target_os = "windows", feature = "asio"))]
        if host.id() != cpal::HostId::Asio {
            return Err("requested ASIO host is unavailable".into());
        }
        #[cfg(not(all(target_os = "windows", feature = "asio")))]
        return Err("ASIO capture requires Windows and the asio feature".into());
    }
    let identifier = super::strip_asio_prefix(identifier);
    let devices: Vec<cpal::Device> = if is_input {
        host.input_devices()
            .map_err(|e| format!("Failed to enumerate input devices: {e}"))?
            .collect()
    } else {
        host.output_devices()
            .map_err(|e| format!("Failed to enumerate output devices: {e}"))?
            .collect()
    };

    // Extract info for matching
    let device_info: Vec<(String, String)> = devices
        .iter()
        .map(|d| {
            let id = d.id().ok().map(|i| i.to_string()).unwrap_or_default();
            let name = d
                .description()
                .ok()
                .map(|desc| desc.name().to_string())
                .unwrap_or_default();
            (id, name)
        })
        .collect();

    if let Some(idx) = match_device_priority(&device_info, identifier) {
        Ok(devices[idx].clone())
    } else {
        Err(format!(
            "{}; use an exact device ID or a unique full name",
            device_not_found_message(identifier, is_input, &device_info)
        ))
    }
}

/// Helpful not-found message listing available devices (shared with fakes).
pub(crate) fn device_not_found_message(
    identifier: &str,
    is_input: bool,
    device_info: &[(String, String)],
) -> String {
    let device_type = if is_input { "input" } else { "output" };
    let available_summary = summarize_available_device_names(device_info, 12);
    format!(
        "Audio device '{}' not found. Available {} devices ({} total): {}",
        identifier,
        device_type,
        device_info.len(),
        available_summary
    )
}
