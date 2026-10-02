//! Immutable microphone calibration data for live level analysis.
//!
//! A profile binds its exact response bytes and RMS pressure anchor to one
//! declared microphone, host, input route, rate, format, gain, and orientation.

// Rust guideline compliant 2026-02-21

use crate::capture_session::CalibrationOrientation;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

const MAX_RESPONSE_BYTES: usize = 1_048_576;
const MAX_RESPONSE_POINTS: usize = 65_536;

/// Identifies the exact machine input path used to establish calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveCalibrationBinding {
    /// User-assigned physical microphone identity; never inferred from a filename.
    pub microphone_id: String,
    /// CPAL host API name, such as `CoreAudio` or `Wasapi`.
    pub host_api: String,
    /// Exact CPAL device ID; a display name alone is not a calibration identity.
    pub input_device_id: String,
    /// Zero-based hardware channel.
    pub input_channel: u16,
    /// Negotiated hardware sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Negotiated CPAL sample format name.
    pub input_sample_format: String,
    /// User-attested fixed input gain in dB; CPAL does not read this back.
    pub declared_input_gain_db: f64,
    /// True only when the operator checked this fixed device gain.
    pub gain_attested: bool,
    /// Microphone orientation associated with the response and anchor.
    pub orientation: CalibrationOrientation,
}

/// Text encoding accepted for a response curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveResponseCurveFormat {
    /// Comma-separated frequency and deviation columns with an optional `frequency_hz,deviation_db` header.
    Csv,
    /// Whitespace-separated frequency and deviation columns.
    Text,
}

/// Response-curve deviation sign convention accepted by live monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveResponseCurveConvention {
    /// The curve is in dB; positive deviation means the microphone is too loud and is subtracted.
    PositiveDbMeansMicrophoneTooLoud,
}

/// Caller-supplied immutable response-curve bytes and optional expected digest.
///
/// Points must have strictly increasing positive frequencies. Deviations are
/// linearly interpolated over log frequency between points, with no extrapolation.
#[derive(Debug, Clone)]
pub struct LiveResponseCurveInput {
    /// Text syntax; column two contains microphone deviation in dB.
    pub format: LiveResponseCurveFormat,
    /// Explicit deviation unit/sign convention; file contents are never guessed.
    pub convention: LiveResponseCurveConvention,
    /// Frequency where deviation is normalized to 0 dB for relative correction, in Hz.
    pub reference_frequency_hz: f64,
    /// Exact bounded bytes parsed and retained by the validated profile.
    pub bytes: Vec<u8>,
    /// Expected SHA-256 hex from the profile manifest, when available.
    pub expected_sha256: Option<String>,
}

/// The supported absolute SPL weighting for a live RMS anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveSplWeighting {
    /// Unweighted/Z-weighted RMS pressure, re 20 µPa.
    Z,
}

/// RMS reference data and the exact machine/curve identities it belongs to.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveRmsSplAnchorInput {
    /// Exact input binding used to make this anchor.
    pub binding: LiveCalibrationBinding,
    /// SHA-256 of the response curve used during anchoring, or `None` without one.
    pub response_sha256: Option<String>,
    /// RMS sample level, in normalized full-scale sample units, for `band_hz`.
    pub reference_rms_full_scale: f64,
    /// External reference-meter level in unweighted dB SPL, re 20 µPa RMS.
    pub reference_level_db_spl: f64,
    /// Reference tone center in Hz.
    pub reference_frequency_hz: f64,
    /// Explicit frequency band measured by the reference meter, in Hz.
    pub band_hz: [f64; 2],
    /// Only unweighted/Z-weighted RMS anchors are accepted by this version.
    pub weighting: LiveSplWeighting,
}

/// Per-machine live analysis settings declared by the operator.
#[derive(Debug, Clone, Serialize)]
pub struct LiveLevelMachineSettings {
    /// User-assigned physical microphone identity.
    pub microphone_id: String,
    /// Fixed input gain as written on or configured for the device; not driver-readback.
    pub declared_input_gain_db: f64,
    /// Confirms the declared gain was checked and will remain fixed during monitoring.
    pub gain_attested: bool,
    /// Microphone orientation for this live measurement.
    pub orientation: CalibrationOrientation,
    /// Optional band for absolute RMS pressure and SPL reporting, in Hz.
    pub spl_band_hz: Option<[f64; 2]>,
}

impl LiveLevelMachineSettings {
    /// Validates machine-specific live calibration declarations.
    ///
    /// # Errors
    /// Rejects an empty microphone identity, nonfinite gain, or malformed SPL band.
    pub fn validate(&self) -> Result<(), String> {
        if self.microphone_id.trim().is_empty() || !self.declared_input_gain_db.is_finite() {
            return Err(
                "live calibration needs a microphone identity and finite declared gain".into(),
            );
        }
        if let Some([low_hz, high_hz]) = self.spl_band_hz
            && (!low_hz.is_finite() || !high_hz.is_finite() || low_hz < 0.0 || high_hz <= low_hz)
        {
            return Err("live SPL band must have finite ascending nonnegative edges".into());
        }
        Ok(())
    }
}

/// Inputs used to build an immutable, validated live calibration profile.
#[derive(Debug, Clone)]
pub struct LiveCalibrationProfileInput {
    /// Machine route and microphone identity for both curve and anchor.
    pub binding: LiveCalibrationBinding,
    /// Optional microphone response curve; omission means no response correction.
    pub response_curve: Option<LiveResponseCurveInput>,
    /// Optional absolute RMS anchor; a response curve by itself is not SPL calibration.
    pub spl_anchor: Option<LiveRmsSplAnchorInput>,
}

/// Validated immutable response and RMS-pressure calibration for one input path.
#[derive(Clone)]
pub struct LiveCalibrationProfile {
    binding: LiveCalibrationBinding,
    response_bytes: Option<Arc<[u8]>>,
    response_sha256: Option<String>,
    response_format: Option<LiveResponseCurveFormat>,
    response_convention: Option<LiveResponseCurveConvention>,
    response_points: Option<Vec<(f64, f64)>>,
    response_reference_frequency_hz: Option<f64>,
    anchor: Option<ValidatedRmsSplAnchor>,
}

impl std::fmt::Debug for LiveCalibrationProfile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LiveCalibrationProfile")
            .field("binding", &self.binding)
            .field("response_sha256", &self.response_sha256)
            .field("response_format", &self.response_format)
            .field("response_convention", &self.response_convention)
            .field(
                "response_point_count",
                &self.response_points.as_ref().map(Vec::len),
            )
            .field(
                "response_reference_frequency_hz",
                &self.response_reference_frequency_hz,
            )
            .field("has_spl_anchor", &self.anchor.is_some())
            .finish()
    }
}

#[derive(Debug, Clone)]
struct ValidatedRmsSplAnchor {
    pressure_scale_squared_pa2: f64,
    reference_frequency_hz: f64,
    band_hz: [f64; 2],
}

/// Machine-readable result of matching a live profile to an opened input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "status", content = "details", rename_all = "snake_case")]
pub enum LiveCalibrationStatus {
    /// No profile was supplied, so the stream contains raw digital measurements only.
    NotConfigured,
    /// A profile was supplied, but the stream ended before its input could be checked.
    NotChecked,
    /// The selected route matches a response curve but has no absolute RMS anchor.
    RelativeOnly,
    /// The selected route matches a response curve and/or RMS pressure anchor.
    Matched {
        /// Whether a validated response curve is active.
        response_curve: bool,
        /// Whether a validated absolute RMS SPL anchor is active.
        spl_anchor: bool,
        /// Input gain is user-declared and is not read back by CPAL.
        gain_basis: LiveGainBasis,
    },
    /// Profile and current machine settings differ; calibrated values are withheld.
    Mismatch(LiveCalibrationMismatch),
}

/// Provenance for the live gain value used to bind a calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveGainBasis {
    /// Gain was explicitly declared by the operator, not verified through CPAL.
    UserDeclared,
}

/// Exact reason a profile does not match the currently opened input path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveCalibrationMismatch {
    /// CPAL could not provide an exact device ID.
    DeviceIdUnavailable,
    /// Physical microphone identity differs.
    MicrophoneId,
    /// CPAL host API differs.
    HostApi,
    /// Exact CPAL device ID differs.
    InputDeviceId,
    /// Hardware channel differs.
    InputChannel,
    /// Negotiated hardware rate differs.
    SampleRate,
    /// Negotiated input PCM format differs.
    SampleFormat,
    /// User-declared fixed input gain differs.
    DeclaredGain,
    /// Operator did not attest that the configured gain was checked and held fixed.
    GainNotAttested,
    /// Microphone orientation differs.
    Orientation,
}

impl LiveCalibrationProfile {
    /// Validates response bytes and an RMS anchor against their immutable machine binding.
    ///
    /// Response curve deviation is positive where the microphone is too loud. The
    /// curve is normalized to the anchor reference frequency before live correction.
    ///
    /// # Errors
    /// Rejects malformed bindings, curves, SHA-256 digests, anchors, or curve/anchor mismatches.
    pub fn new(input: LiveCalibrationProfileInput) -> Result<Self, String> {
        validate_binding(&input.binding)?;
        let response_reference_frequency_hz = input
            .response_curve
            .as_ref()
            .map(|curve| curve.reference_frequency_hz);
        let response_convention = input.response_curve.as_ref().map(|curve| curve.convention);
        let response_format = input.response_curve.as_ref().map(|curve| curve.format);

        let (response_bytes, response_sha256, response_points) = match input.response_curve {
            Some(curve) => {
                if curve.bytes.len() > MAX_RESPONSE_BYTES {
                    return Err("live response curve exceeds the 1 MiB limit".into());
                }
                let digest = sha256_hex(&curve.bytes);
                if let Some(expected) = curve.expected_sha256.as_deref() {
                    validate_sha256(expected)?;
                    if !digest.eq_ignore_ascii_case(expected) {
                        return Err(
                            "live response curve SHA-256 does not match its manifest".into()
                        );
                    }
                }
                let points = parse_response_curve(curve.format, &curve.bytes)?;
                if !curve.reference_frequency_hz.is_finite()
                    || interpolate_deviation(&points, curve.reference_frequency_hz).is_none()
                {
                    return Err("response reference frequency is outside curve coverage".into());
                }
                (
                    Some(Arc::<[u8]>::from(curve.bytes)),
                    Some(digest),
                    Some(points),
                )
            }
            None => (None, None, None),
        };
        if response_bytes.is_none() && input.spl_anchor.is_none() {
            return Err("live calibration profile needs a response curve or RMS SPL anchor".into());
        }

        let anchor = input
            .spl_anchor
            .map(|anchor| {
                validate_anchor(
                    &input.binding,
                    response_sha256.as_deref(),
                    response_points.as_deref(),
                    response_reference_frequency_hz,
                    anchor,
                )
            })
            .transpose()?;

        Ok(Self {
            binding: input.binding,
            response_bytes,
            response_sha256,
            response_format,
            response_convention,
            response_points,
            response_reference_frequency_hz,
            anchor,
        })
    }

    /// Returns the exact hardware and microphone binding.
    pub fn binding(&self) -> &LiveCalibrationBinding {
        &self.binding
    }

    /// Returns the SHA-256 digest of the retained response bytes, when present.
    pub fn response_sha256(&self) -> Option<&str> {
        self.response_sha256.as_deref()
    }

    /// Returns the explicitly selected response-curve sign convention.
    pub fn response_convention(&self) -> Option<LiveResponseCurveConvention> {
        self.response_convention
    }

    /// Returns the explicitly selected response-curve text format.
    pub fn response_format(&self) -> Option<LiveResponseCurveFormat> {
        self.response_format
    }

    /// Returns the frequency where response deviation is normalized to 0 dB.
    pub fn response_reference_frequency_hz(&self) -> Option<f64> {
        self.response_reference_frequency_hz
    }

    /// Returns the retained immutable response-curve bytes, when present.
    pub fn response_bytes(&self) -> Option<&[u8]> {
        self.response_bytes.as_deref()
    }

    /// Reports whether this profile contains a validated absolute RMS anchor.
    pub fn has_spl_anchor(&self) -> bool {
        self.anchor.is_some()
    }

    /// Reports whether this profile has a validated microphone response curve.
    pub fn has_response_curve(&self) -> bool {
        self.response_points.is_some()
    }

    pub(crate) fn match_binding(
        &self,
        current: &LiveCalibrationBinding,
    ) -> Result<(), LiveCalibrationMismatch> {
        if current.input_device_id.is_empty() {
            return Err(LiveCalibrationMismatch::DeviceIdUnavailable);
        }
        if self.binding.microphone_id != current.microphone_id {
            return Err(LiveCalibrationMismatch::MicrophoneId);
        }
        if self.binding.host_api != current.host_api {
            return Err(LiveCalibrationMismatch::HostApi);
        }
        if self.binding.input_device_id != current.input_device_id {
            return Err(LiveCalibrationMismatch::InputDeviceId);
        }
        if self.binding.input_channel != current.input_channel {
            return Err(LiveCalibrationMismatch::InputChannel);
        }
        if self.binding.sample_rate_hz != current.sample_rate_hz {
            return Err(LiveCalibrationMismatch::SampleRate);
        }
        if self.binding.input_sample_format != current.input_sample_format {
            return Err(LiveCalibrationMismatch::SampleFormat);
        }
        if self.binding.declared_input_gain_db.to_bits() != current.declared_input_gain_db.to_bits()
        {
            return Err(LiveCalibrationMismatch::DeclaredGain);
        }
        if !current.gain_attested || self.binding.gain_attested != current.gain_attested {
            return Err(LiveCalibrationMismatch::GainNotAttested);
        }
        if self.binding.orientation != current.orientation {
            return Err(LiveCalibrationMismatch::Orientation);
        }
        Ok(())
    }

    pub(crate) fn response_corrected_psd(
        &self,
        frequencies_hz: &[f64],
        raw_psd: &[f64],
    ) -> Option<Vec<Option<f64>>> {
        let points = self.response_points.as_deref()?;
        if frequencies_hz.len() != raw_psd.len() {
            return None;
        }
        let reference_frequency_hz = self.response_reference_frequency_hz?;
        let reference_deviation_db = interpolate_deviation(points, reference_frequency_hz)?;
        Some(
            frequencies_hz
                .iter()
                .zip(raw_psd)
                .map(|(frequency_hz, power)| {
                    let deviation = interpolate_deviation(points, *frequency_hz)?;
                    let relative_deviation = deviation - reference_deviation_db;
                    let correction = 10.0_f64.powf(-relative_deviation / 10.0);
                    let corrected = *power * correction;
                    (correction.is_finite()
                        && correction > 0.0
                        && corrected.is_finite()
                        && corrected >= 0.0)
                        .then_some(corrected)
                })
                .collect(),
        )
    }

    pub(crate) fn pressure_psd(
        &self,
        frequencies_hz: &[f64],
        raw_psd: &[f64],
    ) -> Option<Vec<Option<f64>>> {
        let anchor = self.anchor.as_ref()?;
        if frequencies_hz.len() != raw_psd.len() {
            return None;
        }
        let reference_deviation_db = match self.response_points.as_deref() {
            Some(points) => Some(interpolate_deviation(
                points,
                anchor.reference_frequency_hz,
            )?),
            None => None,
        };
        Some(
            frequencies_hz
                .iter()
                .zip(raw_psd)
                .map(|(frequency_hz, power)| {
                    let relative_deviation_db = match self.response_points.as_deref() {
                        Some(points) => {
                            let deviation = interpolate_deviation(points, *frequency_hz)?;
                            deviation - reference_deviation_db?
                        }
                        None => {
                            if *frequency_hz < anchor.band_hz[0]
                                || *frequency_hz > anchor.band_hz[1]
                            {
                                return None;
                            }
                            0.0
                        }
                    };
                    let correction = 10.0_f64.powf(-relative_deviation_db / 10.0);
                    let pressure_power = *power * anchor.pressure_scale_squared_pa2 * correction;
                    (correction.is_finite()
                        && correction > 0.0
                        && pressure_power.is_finite()
                        && pressure_power >= 0.0)
                        .then_some(pressure_power)
                })
                .collect(),
        )
    }
}

fn validate_binding(binding: &LiveCalibrationBinding) -> Result<(), String> {
    if !binding.gain_attested {
        return Err(
            "live calibration requires an operator attestation that input gain is fixed".into(),
        );
    }
    if binding.microphone_id.trim().is_empty()
        || binding.host_api.trim().is_empty()
        || binding.input_device_id.trim().is_empty()
        || binding.input_sample_format.trim().is_empty()
        || binding.input_channel >= 64
        || !(8_000..=384_000).contains(&binding.sample_rate_hz)
        || !binding.declared_input_gain_db.is_finite()
    {
        return Err(
            "live calibration binding has an empty identity, invalid route, rate, or gain".into(),
        );
    }
    Ok(())
}

fn validate_anchor(
    binding: &LiveCalibrationBinding,
    response_sha256: Option<&str>,
    response_points: Option<&[(f64, f64)]>,
    response_reference_frequency_hz: Option<f64>,
    anchor: LiveRmsSplAnchorInput,
) -> Result<ValidatedRmsSplAnchor, String> {
    validate_binding(&anchor.binding)?;
    if !same_binding(binding, &anchor.binding) {
        return Err("live SPL anchor binding does not match its calibration profile".into());
    }
    match (response_sha256, anchor.response_sha256.as_deref()) {
        (Some(actual), Some(expected)) if actual.eq_ignore_ascii_case(expected) => {}
        (None, None) => {}
        _ => return Err("live SPL anchor response hash does not match its profile".into()),
    }
    let [low_hz, high_hz] = anchor.band_hz;
    if !anchor.reference_rms_full_scale.is_finite()
        || anchor.reference_rms_full_scale <= 0.0
        || anchor.reference_rms_full_scale >= 1.0
        || !anchor.reference_level_db_spl.is_finite()
        || !anchor.reference_frequency_hz.is_finite()
        || anchor.reference_frequency_hz <= 0.0
        || !low_hz.is_finite()
        || !high_hz.is_finite()
        || low_hz < 0.0
        || high_hz <= low_hz
        || high_hz > f64::from(binding.sample_rate_hz) / 2.0
        || anchor.reference_frequency_hz < low_hz
        || anchor.reference_frequency_hz > high_hz
    {
        return Err("live RMS SPL anchor has invalid RMS, level, frequency, or band".into());
    }
    if let Some(points) = response_points
        && interpolate_deviation(points, anchor.reference_frequency_hz).is_none()
    {
        return Err("live SPL anchor reference frequency is outside response coverage".into());
    }
    if let Some(response_reference_frequency_hz) = response_reference_frequency_hz
        && response_reference_frequency_hz.to_bits() != anchor.reference_frequency_hz.to_bits()
    {
        return Err("live response and RMS anchor reference frequencies must match exactly".into());
    }
    // Convert unweighted RMS dB SPL re 20 µPa to a Pa-per-FS-RMS scale.
    let reference_pressure_pa = 20.0e-6 * 10.0_f64.powf(anchor.reference_level_db_spl / 20.0);
    let pressure_per_full_scale_rms_pa = reference_pressure_pa / anchor.reference_rms_full_scale;
    let pressure_scale_squared_pa2 = pressure_per_full_scale_rms_pa.powi(2);
    if !pressure_per_full_scale_rms_pa.is_finite()
        || pressure_per_full_scale_rms_pa <= 0.0
        || !pressure_scale_squared_pa2.is_finite()
        || pressure_scale_squared_pa2 <= 0.0
    {
        return Err("live RMS SPL anchor produces an invalid pressure scale".into());
    }
    Ok(ValidatedRmsSplAnchor {
        pressure_scale_squared_pa2,
        reference_frequency_hz: anchor.reference_frequency_hz,
        band_hz: anchor.band_hz,
    })
}

fn same_binding(left: &LiveCalibrationBinding, right: &LiveCalibrationBinding) -> bool {
    left.microphone_id == right.microphone_id
        && left.host_api == right.host_api
        && left.input_device_id == right.input_device_id
        && left.input_channel == right.input_channel
        && left.sample_rate_hz == right.sample_rate_hz
        && left.input_sample_format == right.input_sample_format
        && left.declared_input_gain_db.to_bits() == right.declared_input_gain_db.to_bits()
        && left.gain_attested == right.gain_attested
        && left.orientation == right.orientation
}

fn parse_response_curve(
    format: LiveResponseCurveFormat,
    bytes: &[u8],
) -> Result<Vec<(f64, f64)>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("response curve is not UTF-8: {error}"))?;
    let mut points = Vec::new();
    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let columns: Vec<_> = match format {
            LiveResponseCurveFormat::Csv => line.split(',').map(str::trim).collect(),
            LiveResponseCurveFormat::Text => line.split_whitespace().collect(),
        };
        if columns.len() != 2 {
            return Err(format!(
                "response curve line {} must have exactly two columns",
                line_index + 1
            ));
        }
        if format == LiveResponseCurveFormat::Csv
            && columns[0].eq_ignore_ascii_case("frequency_hz")
            && columns[1].eq_ignore_ascii_case("deviation_db")
        {
            if !points.is_empty() {
                return Err(format!(
                    "response curve header on line {} must precede data",
                    line_index + 1
                ));
            }
            continue;
        }
        let frequency_hz = columns[0].parse::<f64>().map_err(|error| {
            format!(
                "response curve line {} has an invalid frequency: {error}",
                line_index + 1
            )
        })?;
        let deviation_db = columns[1].parse::<f64>().map_err(|error| {
            format!(
                "response curve line {} has an invalid deviation: {error}",
                line_index + 1
            )
        })?;
        if !frequency_hz.is_finite()
            || frequency_hz <= 0.0
            || !deviation_db.is_finite()
            || points
                .last()
                .is_some_and(|(previous, _)| frequency_hz <= *previous)
        {
            return Err(format!(
                "response curve line {} must have finite deviations and strictly increasing positive frequencies",
                line_index + 1
            ));
        }
        points.push((frequency_hz, deviation_db));
        if points.len() > MAX_RESPONSE_POINTS {
            return Err("live response curve exceeds the 65536-point limit".into());
        }
    }
    if points.len() < 2 {
        return Err("live response curve needs at least two frequency points".into());
    }
    Ok(points)
}

fn interpolate_deviation(points: &[(f64, f64)], frequency_hz: f64) -> Option<f64> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return None;
    }
    let first = points.first()?;
    let last = points.last()?;
    if frequency_hz < first.0 || frequency_hz > last.0 {
        return None;
    }
    let index = points.partition_point(|(frequency, _)| *frequency < frequency_hz);
    if let Some((exact_frequency, deviation)) = points.get(index)
        && *exact_frequency == frequency_hz
    {
        return Some(*deviation);
    }
    if index == 0 || index >= points.len() {
        return None;
    }
    let (low_frequency, low_deviation) = points[index - 1];
    let (high_frequency, high_deviation) = points[index];
    let fraction =
        (frequency_hz.ln() - low_frequency.ln()) / (high_frequency.ln() - low_frequency.ln());
    let deviation = low_deviation + fraction * (high_deviation - low_deviation);
    deviation.is_finite().then_some(deviation)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("live response curve SHA-256 must contain 64 hexadecimal characters".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> LiveCalibrationBinding {
        LiveCalibrationBinding {
            microphone_id: "room-mic-a".into(),
            host_api: "CoreAudio".into(),
            input_device_id: "coreaudio:input:42".into(),
            input_channel: 1,
            sample_rate_hz: 48_000,
            input_sample_format: "F32".into(),
            declared_input_gain_db: 12.0,
            gain_attested: true,
            orientation: CalibrationOrientation::OnAxis,
        }
    }

    fn curve(bytes: &[u8]) -> LiveResponseCurveInput {
        LiveResponseCurveInput {
            format: LiveResponseCurveFormat::Csv,
            convention: LiveResponseCurveConvention::PositiveDbMeansMicrophoneTooLoud,
            reference_frequency_hz: 1_000.0,
            bytes: bytes.to_vec(),
            expected_sha256: Some(sha256_hex(bytes)),
        }
    }

    fn anchor(
        binding: &LiveCalibrationBinding,
        response_sha256: Option<String>,
    ) -> LiveRmsSplAnchorInput {
        LiveRmsSplAnchorInput {
            binding: binding.clone(),
            response_sha256,
            reference_rms_full_scale: 0.1,
            reference_level_db_spl: 20.0 * (1.0_f64 / 20.0e-6).log10(),
            reference_frequency_hz: 1_000.0,
            band_hz: [800.0, 1_200.0],
            weighting: LiveSplWeighting::Z,
        }
    }

    #[test]
    fn response_curve_is_hash_bound_and_positive_deviation_is_subtracted_once() {
        let response_bytes = b"frequency_hz,deviation_db\n100,2\n1000,4\n10000,10\n";
        let profile = LiveCalibrationProfile::new(LiveCalibrationProfileInput {
            binding: binding(),
            response_curve: Some(curve(response_bytes)),
            spl_anchor: None,
        })
        .unwrap();
        assert_eq!(profile.response_bytes(), Some(response_bytes.as_slice()));
        assert_eq!(
            profile.response_sha256(),
            Some(sha256_hex(response_bytes).as_str())
        );
        assert_eq!(
            profile.response_format(),
            Some(LiveResponseCurveFormat::Csv)
        );
        assert_eq!(
            profile.response_convention(),
            Some(LiveResponseCurveConvention::PositiveDbMeansMicrophoneTooLoud)
        );
        assert_eq!(profile.response_reference_frequency_hz(), Some(1_000.0));

        let corrected = profile
            .response_corrected_psd(&[100.0, 1_000.0, 10_000.0, 20_000.0], &[1.0; 4])
            .unwrap();
        assert!((corrected[0].unwrap() - 10.0_f64.powf(2.0 / 10.0)).abs() < 1e-12);
        assert!((corrected[1].unwrap() - 1.0).abs() < 1e-12);
        assert!((corrected[2].unwrap() - 10.0_f64.powf(-6.0 / 10.0)).abs() < 1e-12);
        assert!(corrected[3].is_none());
    }

    #[test]
    fn pressure_scale_is_bound_to_exact_curve_and_reference_frequency() {
        let response_bytes = b"100,2\n1000,4\n10000,10\n";
        let binding = binding();
        let profile = LiveCalibrationProfile::new(LiveCalibrationProfileInput {
            binding: binding.clone(),
            response_curve: Some(curve(response_bytes)),
            spl_anchor: Some(anchor(&binding, Some(sha256_hex(response_bytes)))),
        })
        .unwrap();
        let pressure_psd = profile
            .pressure_psd(&[1_000.0, 10_000.0, 20_000.0], &[0.01; 3])
            .unwrap();
        assert!((pressure_psd[0].unwrap() - 1.0).abs() < 1e-10);
        assert!((pressure_psd[1].unwrap() - 10.0_f64.powf(-6.0 / 10.0)).abs() < 1e-10);
        assert!(pressure_psd[2].is_none());
        assert!(profile.has_spl_anchor());
    }

    #[test]
    fn invalid_digest_binding_or_gain_attestation_refuses_calibrated_use() {
        let response_bytes = b"100,2\n1000,4\n10000,10\n";
        let mut invalid_digest = curve(response_bytes);
        invalid_digest.expected_sha256 = Some("0".repeat(64));
        assert!(
            LiveCalibrationProfile::new(LiveCalibrationProfileInput {
                binding: binding(),
                response_curve: Some(invalid_digest),
                spl_anchor: None,
            })
            .is_err()
        );

        let binding = binding();
        assert!(
            LiveCalibrationProfile::new(LiveCalibrationProfileInput {
                binding: binding.clone(),
                response_curve: Some(curve(response_bytes)),
                spl_anchor: Some(anchor(&binding, Some("f".repeat(64)))),
            })
            .is_err()
        );

        let profile = LiveCalibrationProfile::new(LiveCalibrationProfileInput {
            binding: binding.clone(),
            response_curve: Some(curve(response_bytes)),
            spl_anchor: None,
        })
        .unwrap();
        let mut current = binding;
        current.gain_attested = false;
        assert_eq!(
            profile.match_binding(&current),
            Err(LiveCalibrationMismatch::GainNotAttested)
        );
    }

    #[test]
    fn response_curve_validation_rejects_unordered_and_out_of_range_reference_data() {
        let binding = binding();
        let mut unordered = curve(b"1000,0\n500,1\n");
        unordered.reference_frequency_hz = 750.0;
        assert!(
            LiveCalibrationProfile::new(LiveCalibrationProfileInput {
                binding: binding.clone(),
                response_curve: Some(unordered),
                spl_anchor: None,
            })
            .is_err()
        );

        let mut out_of_range = curve(b"100,0\n1000,1\n");
        out_of_range.reference_frequency_hz = 20.0;
        assert!(
            LiveCalibrationProfile::new(LiveCalibrationProfileInput {
                binding,
                response_curve: Some(out_of_range),
                spl_anchor: None,
            })
            .is_err()
        );
    }

    #[test]
    fn response_parser_rejects_ambiguous_headers_and_nonfinite_interpolation() {
        let binding = binding();
        let mut ambiguous_header = LiveResponseCurveInput {
            format: LiveResponseCurveFormat::Csv,
            convention: LiveResponseCurveConvention::PositiveDbMeansMicrophoneTooLoud,
            reference_frequency_hz: 1_000.0,
            bytes: b"frequency_khz,deviation_db\n100,0\n1000,0\n".to_vec(),
            expected_sha256: None,
        };
        assert!(
            LiveCalibrationProfile::new(LiveCalibrationProfileInput {
                binding: binding.clone(),
                response_curve: Some(ambiguous_header.clone()),
                spl_anchor: None,
            })
            .is_err()
        );

        ambiguous_header.bytes = b"100,0,extra\n1000,0,extra\n".to_vec();
        assert!(
            LiveCalibrationProfile::new(LiveCalibrationProfileInput {
                binding: binding.clone(),
                response_curve: Some(ambiguous_header),
                spl_anchor: None,
            })
            .is_err()
        );

        let mut overflow_curve = curve(b"100,1e308\n1000,-1e308\n");
        overflow_curve.reference_frequency_hz = 316.227_766_016_837_96;
        assert!(
            LiveCalibrationProfile::new(LiveCalibrationProfileInput {
                binding,
                response_curve: Some(overflow_curve),
                spl_anchor: None,
            })
            .is_err()
        );
    }

    #[test]
    fn rms_anchor_rejects_a_pressure_scale_whose_square_overflows() {
        let binding = binding();
        let mut invalid_anchor = anchor(&binding, None);
        invalid_anchor.reference_rms_full_scale = 1.0e-308;
        assert!(
            LiveCalibrationProfile::new(LiveCalibrationProfileInput {
                binding,
                response_curve: None,
                spl_anchor: Some(invalid_anchor),
            })
            .is_err()
        );
    }

    #[test]
    fn correction_factors_that_underflow_are_unknown_bins() {
        let response_bytes = b"100,1e308\n1000,-1e308\n10000,-1e308\n";
        let initial_binding = binding();
        let profile = LiveCalibrationProfile::new(LiveCalibrationProfileInput {
            binding: initial_binding.clone(),
            response_curve: Some(curve(response_bytes)),
            spl_anchor: Some(anchor(&initial_binding, Some(sha256_hex(response_bytes)))),
        })
        .unwrap();

        let corrected = profile
            .response_corrected_psd(&[100.0, 1_000.0, 10_000.0], &[1.0; 3])
            .unwrap();
        assert!(corrected[0].is_none());
        assert!(corrected[1].is_some());
        let pressure = profile
            .pressure_psd(&[100.0, 1_000.0, 10_000.0], &[1.0; 3])
            .unwrap();
        assert!(pressure[0].is_none());
        assert!(pressure[1].is_some());

        let response_bytes = b"100,-1e308\n1000,1e308\n10000,1e308\n";
        let binding = binding();
        let overflow_profile = LiveCalibrationProfile::new(LiveCalibrationProfileInput {
            binding: binding.clone(),
            response_curve: Some(curve(response_bytes)),
            spl_anchor: Some(anchor(&binding, Some(sha256_hex(response_bytes)))),
        })
        .unwrap();
        let corrected = overflow_profile
            .response_corrected_psd(&[100.0, 1_000.0, 10_000.0], &[1.0; 3])
            .unwrap();
        assert!(corrected[0].is_none());
        let pressure = overflow_profile
            .pressure_psd(&[100.0, 1_000.0, 10_000.0], &[1.0; 3])
            .unwrap();
        assert!(pressure[0].is_none());
    }

    #[test]
    fn rms_anchor_rejects_a_pressure_scale_whose_square_underflows() {
        let binding = binding();
        let mut invalid_anchor = anchor(&binding, None);
        invalid_anchor.reference_level_db_spl = -4_000.0;
        assert!(
            LiveCalibrationProfile::new(LiveCalibrationProfileInput {
                binding,
                response_curve: None,
                spl_anchor: Some(invalid_anchor),
            })
            .is_err()
        );
    }
}
