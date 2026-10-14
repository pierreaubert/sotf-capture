//! Shared configuration and preflight validation for multi-microphone capture.
//!
//! Validation freezes the declared geometry, gain and calibration assignments.
//! Device negotiation and take QA are separate steps: a valid plan is not evidence
//! of a synchronized recording or permission to render coherent measurements.

use serde::{Deserialize, Serialize};
#[cfg(not(target_os = "ios"))]
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::PathBuf;

#[cfg(not(target_os = "ios"))]
pub mod analysis;
#[cfg(not(target_os = "ios"))]
pub mod clock;
#[cfg(not(target_os = "ios"))]
mod coherence;
#[cfg(not(target_os = "ios"))]
pub mod ctc;
#[cfg(not(target_os = "ios"))]
pub mod handoff;
#[cfg(not(target_os = "ios"))]
pub mod manifest;
#[cfg(not(target_os = "ios"))]
mod phase;
pub mod protocol;
#[cfg(not(target_os = "ios"))]
pub mod record;
#[cfg(not(target_os = "ios"))]
pub mod reflections;

#[cfg(not(target_os = "ios"))]
pub(super) fn digest_lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut hex, "{byte:02x}").expect("writing to a String cannot fail");
    }
    hex
}

#[cfg(not(target_os = "ios"))]
pub(super) fn calibration_sha256(bytes: &[u8]) -> String {
    digest_lower_hex(Sha256::digest(bytes).as_ref())
}

#[cfg(all(test, not(target_os = "ios")))]
mod hash_tests {
    use super::calibration_sha256;

    #[test]
    fn calibration_hash_is_lowercase_sha256() {
        assert_eq!(
            calibration_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}

/// Spatial interpretation fixed for every take in a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureGeometry {
    /// Independent seats for spatial magnitude coverage.
    Spread,
    /// A measured microphone cluster for conditional direction estimation.
    Compact,
}

/// Calibration orientation relative to the capsule axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationOrientation {
    /// Capsule aimed at the source.
    OnAxis,
    /// Capsule perpendicular to the source direction.
    NinetyDegrees,
}

/// One physical microphone and its fixed session settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureMicrophone {
    /// Unique identity used in take provenance, independent of device ordering.
    pub id: String,
    /// Exact audio input device ID or unique name; never silently use the default.
    pub device: String,
    /// Zero-based channel within the selected input device.
    pub input_channel: u16,
    /// Individual microphone calibration file, resolved relative to the plan.
    pub calibration_file: PathBuf,
    /// Orientation for which the assigned calibration was measured.
    pub calibration_orientation: CalibrationOrientation,
    /// Documented input gain, held fixed throughout the session.
    pub gain_db: f64,
    /// Microphone coordinates in meters in one session coordinate system.
    pub position_m: [f64; 3],
    /// Survey uncertainty in millimeters, required for compact-array use.
    pub position_uncertainty_mm: f64,
}

/// A loudspeaker excited separately while every microphone records.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSource {
    /// Unique source label used to associate all simultaneous takes.
    pub id: String,
    /// Zero-based hardware output channel.
    pub output_channel: u16,
}

/// A fixed acoustic timing emitter surveyed in the microphone coordinate frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureTimingReference {
    /// Hardware channel used for both timing chirps on every source take.
    pub output_channel: u16,
    /// Timing emitter acoustic center, in meters.
    pub position_m: [f64; 3],
    /// Upper bound on emitter-position error, in millimeters.
    pub position_uncertainty_mm: f64,
    /// Measured or estimated speed of sound in meters per second.
    pub sound_speed_m_s: f64,
    /// Upper bound on sound-speed error in meters per second.
    pub sound_speed_uncertainty_m_s: f64,
}

/// Physical room dimensions in meters for `RoomEQ` Schroeder handling.
///
/// Length, width, and height are all required when present. Values are
/// validated on plan load; `RoomEQ` resolves them into engine settings and
/// falls back to its default Schroeder frequency when absent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoomDimensions {
    /// Room length in meters, finite and positive.
    pub length: f64,
    /// Room width in meters, finite and positive.
    pub width: f64,
    /// Room height in meters, finite and positive.
    pub height: f64,
}

/// Explicit logarithmic sweep settings shared by all sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSweep {
    /// Nominal sweep duration, excluding timing chirps and silence, in
    /// seconds. With bass weighting this is the minimum; sub-100 Hz
    /// octaves extend past it and the journal records the actual span.
    pub duration_secs: f64,
    /// Low sweep frequency in Hz.
    pub start_hz: f64,
    /// High sweep frequency in Hz, strictly below Nyquist.
    pub end_hz: f64,
    /// Linear playback peak amplitude, in (0, 1].
    pub amplitude: f64,
    /// Seconds per octave below 100 Hz, within [1, 10]; absent keeps the
    /// legacy plain fixed-duration log sweep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_octave_duration_s: Option<f32>,
}

/// Delay-probe pass parameters, played once per source after the sweeps.
///
/// Probes run through the plan output device while the first microphone
/// records; results land in the raw journal for `RoomEQ` delay evidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureProbe {
    /// Narrowband probe duration per channel in milliseconds.
    pub probe_duration_ms: f32,
    /// Silence gap between channel probes in milliseconds.
    pub silence_duration_ms: f32,
    /// Probe playback level in dBFS, within [-60, 0].
    pub signal_level_db: f32,
}

/// Bass-anchor pass parameters: a steady low-frequency tone per source.
///
/// The lock-in phase at `bass_freq_hz` anchors low-frequency timing
/// confidence. A plan loopback on the probe microphone's device turns the
/// reported phase into loopback-corrected acoustic phase.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureBassAnchor {
    /// Steady-state tone centre frequency in Hz, below Nyquist.
    pub bass_freq_hz: f32,
    /// Total tone length in seconds, steady portion plus fades.
    pub bass_duration_s: f32,
    /// Half-Hann fade-in/out length in milliseconds.
    pub fade_ms: f32,
    /// Lock-in sub-window count for the stability estimate, at least one.
    pub num_windows: u16,
    /// Silence gap between channel tones in milliseconds.
    pub silence_duration_ms: f32,
    /// Tone playback level in dBFS, within [-60, 0].
    pub signal_level_db: f32,
}

/// Electrical loopback input recorded alongside acoustic takes.
///
/// Shared by the bass-anchor pass (same device as the probe microphone)
/// and later CTC raw-sweep takes. Never a microphone: no calibration or
/// position applies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureLoopback {
    /// Exact audio input device ID or unique name.
    pub device: String,
    /// Zero-based channel within the selected input device.
    pub input_channel: u16,
}

/// Binaural head position for one CTC capture session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureHeadPosition {
    /// Stable identity shared across the merged head-position matrix.
    pub id: String,
    /// Head x/y/z in meters in the session coordinate frame.
    pub x: f64,
    /// Head y in meters (see [`CaptureHeadPosition::x`]).
    pub y: f64,
    /// Head z in meters (see [`CaptureHeadPosition::x`]).
    pub z: f64,
    /// Head yaw in degrees, 0 facing the reference axis.
    pub yaw_deg: f64,
}

/// Crosstalk-cancellation capture declaration for one head position.
///
/// Names the simultaneously recorded ear-microphone pair. The plan loopback
/// is recorded with every sweep take and backs `RoomEQ` loopback alignment;
/// each head position is one session, merged later by `plan-ctc-merge`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureCtc {
    /// Microphone identity recording the left ear (stereo channel 1).
    pub left_ear: String,
    /// Microphone identity recording the right ear (stereo channel 2).
    pub right_ear: String,
    /// Head position of this session.
    pub head_position: CaptureHeadPosition,
}

/// Serializable multi-microphone session declaration.
///
/// The identical start/end timing chirps are the drift path: per-take
/// clock skew is estimated from the chirp pair and corrected before
/// analysis. Chirpless takes have no unbiased drift source and skip
/// drift estimation with `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSessionPlan {
    /// Schema version; currently 1.
    pub version: u32,
    /// One geometry mode applies to the entire session.
    pub geometry: CaptureGeometry,
    /// Requested common nominal rate; all devices must support it exactly.
    pub sample_rate_hz: u32,
    /// Explicit output device selector.
    pub output_device: String,
    /// Two to four simultaneously recorded microphones.
    pub microphones: Vec<CaptureMicrophone>,
    /// Sources played sequentially in this order.
    pub sources: Vec<CaptureSource>,
    /// Number of independent acquisitions for every source/microphone pair.
    /// Every repeat remains a separate take; measurements are never averaged.
    #[serde(default = "default_repeat_count")]
    pub repeat_count: u32,
    /// Sweep parameters fixed throughout the session.
    pub sweep: CaptureSweep,
    /// Fixed timing emitter and geometry; absent geometry disables coherent use.
    /// Without this declaration the first source channel emits timing chirps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing_reference: Option<CaptureTimingReference>,
    /// Physical room dimensions in meters, projected to `RoomEQ` when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub room_dimensions: Option<RoomDimensions>,
    /// Free-form room treatment, seating, and placement notes for `RoomEQ`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup_description: Option<String>,
    /// Delay-probe pass after the sweeps; absent runs sweeps only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe: Option<CaptureProbe>,
    /// Bass-anchor pass after the sweeps; absent runs sweeps only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_anchor: Option<CaptureBassAnchor>,
    /// Electrical loopback input for anchor/CTC passes; never a microphone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loopback: Option<CaptureLoopback>,
    /// CTC ear-pair capture for one head position; absent skips CTC.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctc: Option<CaptureCtc>,
}

fn default_repeat_count() -> u32 {
    1
}

/// Validated settings that cannot be changed during capture.
#[derive(Debug, Clone)]
pub struct ValidatedCaptureSession(CaptureSessionPlan);

/// A session declaration rejected before opening audio devices.
#[derive(Debug, thiserror::Error)]
#[error("invalid capture session: {0}")]
pub struct CapturePlanError(String);

impl CaptureSessionPlan {
    /// Validate and freeze all declared session settings.
    ///
    /// Compact sessions require positions known within one millimeter. This is
    /// only a survey precondition: aperture, frequency, clock uncertainty and
    /// TDOA consistency still determine whether individual directions are usable.
    ///
    /// # Errors
    /// Rejects unsupported versions, invalid rates/sweeps, duplicate identities
    /// or routes, non-finite settings, missing calibration, and invalid geometry.
    #[expect(
        clippy::too_many_lines,
        reason = "Keep the related validation rules together before any audio acquisition can begin."
    )]
    #[expect(
        clippy::float_cmp,
        reason = "Exactly duplicated microphone coordinates are invalid; nearby distinct positions remain permitted."
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Checking the f32 cast detects amplitude underflow at the waveform boundary."
    )]
    pub fn validate(self) -> Result<ValidatedCaptureSession, CapturePlanError> {
        let reject = |message: &str| CapturePlanError(message.to_owned());
        if self.version != 1 {
            return Err(reject("unsupported plan version (expected 1)"));
        }
        // The timing chirp occupies 2–8 kHz; Nyquist must exceed its upper edge.
        if !(16_001..=384_000).contains(&self.sample_rate_hz) {
            return Err(reject(
                "sample rate must be in 16001..=384000 Hz for capture",
            ));
        }
        if self.output_device.trim().is_empty() {
            return Err(reject("an explicit output device is required"));
        }
        if !(1..=16).contains(&self.repeat_count) {
            return Err(reject("repeat count must be in 1..=16"));
        }
        if !(2..=4).contains(&self.microphones.len()) {
            return Err(reject("a session requires two to four microphones"));
        }
        let mut ids = HashSet::new();
        let mut routes = HashSet::new();
        for mic in &self.microphones {
            if mic.id.trim().is_empty() || !ids.insert(&mic.id) {
                return Err(reject("microphone identities must be nonempty and unique"));
            }
            if mic.device.trim().is_empty()
                || mic.input_channel >= 64
                || !routes.insert((&mic.device, mic.input_channel))
            {
                return Err(reject(
                    "microphone device/channel routes must be explicit, unique, and below channel 64",
                ));
            }
            if mic.calibration_file.as_os_str().is_empty() {
                return Err(reject("each microphone requires its own calibration file"));
            }
            if !mic.gain_db.is_finite()
                || mic.position_m.iter().any(|value| !value.is_finite())
                || !mic.position_uncertainty_mm.is_finite()
                || mic.position_uncertainty_mm < 0.0
            {
                return Err(reject(
                    "microphone gain and geometry must be finite and uncertainty nonnegative",
                ));
            }
            if self.geometry == CaptureGeometry::Compact && mic.position_uncertainty_mm > 1.0 {
                return Err(reject(
                    "compact microphone positions require at most 1 mm uncertainty",
                ));
            }
        }
        if self.geometry == CaptureGeometry::Compact {
            for (index, mic) in self.microphones.iter().enumerate() {
                if self.microphones[..index]
                    .iter()
                    .any(|other| other.position_m == mic.position_m)
                {
                    return Err(reject("compact microphones must occupy distinct positions"));
                }
            }
        }
        ids.clear();
        let mut outputs = HashSet::new();
        if self.sources.is_empty() {
            return Err(reject("at least one source is required"));
        }
        for source in &self.sources {
            if source.id.trim().is_empty()
                || source.output_channel >= 64
                || !ids.insert(&source.id)
                || !outputs.insert(source.output_channel)
            {
                return Err(reject(
                    "source identities must be nonempty and unique, with unique output channels below 64",
                ));
            }
        }
        let take_count = self
            .sources
            .len()
            .checked_mul(self.microphones.len())
            .and_then(|count| count.checked_mul(self.repeat_count as usize))
            .ok_or_else(|| reject("capture take count overflow"))?;
        if take_count > 1024 {
            return Err(reject(
                "capture session exceeds the 1024-take metadata limit",
            ));
        }
        if let Some(room) = &self.room_dimensions {
            let volume = room.length * room.width * room.height;
            if ![room.length, room.width, room.height, volume]
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            {
                return Err(reject(
                    "room dimensions require finite positive length, width, height, and volume in meters",
                ));
            }
        }
        if self
            .setup_description
            .as_deref()
            .is_some_and(|note| note.trim().is_empty())
        {
            return Err(reject("setup description must be nonempty when supplied"));
        }
        if let Some(probe) = &self.probe {
            if !probe.probe_duration_ms.is_finite()
                || probe.probe_duration_ms <= 0.0
                || !probe.silence_duration_ms.is_finite()
                || probe.silence_duration_ms < 0.0
                || !probe.signal_level_db.is_finite()
                || !(-60.0..=0.0).contains(&probe.signal_level_db)
            {
                return Err(reject(
                    "probe requires finite positive duration, nonnegative silence, and level in [-60, 0] dBFS",
                ));
            }
        }
        if let Some(anchor) = &self.bass_anchor {
            let nyquist = f64::from(self.sample_rate_hz) / 2.0;
            if !anchor.bass_freq_hz.is_finite()
                || anchor.bass_freq_hz <= 0.0
                || f64::from(anchor.bass_freq_hz) >= nyquist
                || !anchor.bass_duration_s.is_finite()
                || anchor.bass_duration_s <= 0.0
                || !anchor.fade_ms.is_finite()
                || anchor.fade_ms < 0.0
                || f64::from(anchor.fade_ms) * 2.0 / 1000.0 > f64::from(anchor.bass_duration_s)
                || anchor.num_windows == 0
                || !anchor.silence_duration_ms.is_finite()
                || anchor.silence_duration_ms < 0.0
                || !anchor.signal_level_db.is_finite()
                || !(-60.0..=0.0).contains(&anchor.signal_level_db)
            {
                return Err(reject(
                    "bass anchor requires a sub-Nyquist tone, fades within its duration, at least one window, nonnegative silence, and level in [-60, 0] dBFS",
                ));
            }
        }
        if let Some(loopback) = &self.loopback {
            if loopback.device.trim().is_empty()
                || loopback.input_channel >= 64
                || self.microphones.iter().any(|mic| {
                    mic.device == loopback.device && mic.input_channel == loopback.input_channel
                })
            {
                return Err(reject(
                    "loopback route must be explicit, below channel 64, and distinct from every microphone route",
                ));
            }
            if self.bass_anchor.is_some() && loopback.device != self.microphones[0].device {
                return Err(reject(
                    "bass-anchor loopback must share the probe microphone's device",
                ));
            }
        }
        if let Some(ctc) = &self.ctc {
            let ears_known = |id: &str| self.microphones.iter().any(|mic| mic.id == id);
            let head = &ctc.head_position;
            if ctc.left_ear == ctc.right_ear
                || !ears_known(&ctc.left_ear)
                || !ears_known(&ctc.right_ear)
                || head.id.trim().is_empty()
                || ![head.x, head.y, head.z, head.yaw_deg]
                    .iter()
                    .all(|value| value.is_finite())
            {
                return Err(reject(
                    "ctc requires two distinct known ear microphones and a finite head position with a nonempty id",
                ));
            }
            if self.loopback.is_none() {
                return Err(reject("ctc requires a plan loopback route"));
            }
        }
        let sweep = &self.sweep;
        if let Some(reference) = &self.timing_reference {
            if reference.output_channel >= 64
                || reference.position_m.iter().any(|v| !v.is_finite())
                || !reference.position_uncertainty_mm.is_finite()
                || reference.position_uncertainty_mm < 0.0
                || !reference.sound_speed_m_s.is_finite()
                || !(250.0..=400.0).contains(&reference.sound_speed_m_s)
                || !reference.sound_speed_uncertainty_m_s.is_finite()
                || reference.sound_speed_uncertainty_m_s < 0.0
                || reference.sound_speed_uncertainty_m_s >= reference.sound_speed_m_s
            {
                return Err(reject(
                    "invalid timing-reference route, geometry, or sound-speed uncertainty",
                ));
            }
        }
        if !sweep.duration_secs.is_finite()
            || sweep.duration_secs <= 0.0
            || !sweep.start_hz.is_finite()
            || sweep.start_hz <= 0.0
            || !sweep.end_hz.is_finite()
            || sweep.end_hz <= sweep.start_hz
            || sweep.end_hz >= f64::from(self.sample_rate_hz) / 2.0
            || !sweep.amplitude.is_finite()
            || sweep.amplitude <= 0.0
            || sweep.amplitude > 1.0
        {
            return Err(reject(
                "sweep requires finite positive duration, ascending frequencies below Nyquist, and amplitude in (0, 1]",
            ));
        }
        if sweep
            .bass_octave_duration_s
            .is_some_and(|bass| !bass.is_finite() || !(1.0..=10.0).contains(&bass))
        {
            return Err(reject(
                "bass octave duration must be finite and within [1, 10] seconds",
            ));
        }
        // Budget includes chirps, decay gaps and engine startup/drain storage.
        // Match the engine's bounded per-microphone capture memory budget.
        // Bass weighting extends the sweep past its nominal duration; the
        // bound prices the weighted total, not the nominal one.
        let sweep_seconds = octave_weighted_sweep_seconds(
            sweep.start_hz,
            sweep.end_hz,
            sweep.bass_octave_duration_s,
            sweep.duration_secs,
        );
        if (sweep_seconds + 14.0) * f64::from(self.sample_rate_hz) > 16_000_000.0 {
            return Err(reject(
                "sweep duration exceeds bounded capture storage at this sample rate",
            ));
        }
        if sweep.amplitude as f32 == 0.0
            || sweep.duration_secs * f64::from(self.sample_rate_hz) < 2.0
        {
            return Err(reject(
                "sweep must have at least two samples and a representable nonzero amplitude",
            ));
        }
        Ok(ValidatedCaptureSession(self))
    }
}

impl ValidatedCaptureSession {
    /// Inspect the frozen session configuration.
    #[must_use]
    pub fn plan(&self) -> &CaptureSessionPlan {
        &self.0
    }
}

/// Nominal sweep seconds including bass weighting, if any.
///
/// Mirrors the octave-scaled generator's timing law (full dwell below
/// 100 Hz, half to 1 kHz, quarter above, floored at the nominal duration)
/// so validation prices the played span without generating audio. Unclamped
/// start frequencies only widen the bound, never narrow it.
pub(crate) fn octave_weighted_sweep_seconds(
    start_hz: f64,
    end_hz: f64,
    bass_octave_duration_s: Option<f32>,
    nominal_secs: f64,
) -> f64 {
    let Some(bass) = bass_octave_duration_s.map(f64::from) else {
        return nominal_secs;
    };
    let zone = |lo: f64, hi: f64| {
        if hi > lo { (hi / lo).log2() } else { 0.0 }
    };
    let weighted = zone(start_hz, 100.0f64.min(end_hz)) * bass
        + zone(100.0f64.max(start_hz), 1000.0f64.min(end_hz)) * bass * 0.5
        + zone(1000.0f64.max(start_hz), end_hz) * bass * 0.25;
    weighted.max(nominal_secs)
}

/// Output channel emitting timing chirps on every source take.
///
/// A declared fixed emitter wins; otherwise the first source channel
/// carries the chirps and takes stay arrival-aligned.
#[must_use]
pub fn timing_emitter_channel(session: &ValidatedCaptureSession) -> u16 {
    session
        .plan()
        .timing_reference
        .as_ref()
        .map_or(session.plan().sources[0].output_channel, |reference| {
            reference.output_channel
        })
}

/// One-line timing-emitter declaration for validate/record reporting.
///
/// Names the chirp channel and whether a fixed emitter survey backs
/// coherent use. Per-take reference identities resolve at record time.
#[must_use]
pub fn timing_emitter_summary(session: &ValidatedCaptureSession) -> String {
    let channel = timing_emitter_channel(session);
    if let Some(emitter) = &session.plan().timing_reference {
        format!(
            "timing chirps on output channel {channel} (dedicated emitter at [{:.3}, {:.3}, {:.3}] m); per-take reference identity resolves at record time",
            emitter.position_m[0], emitter.position_m[1], emitter.position_m[2]
        )
    } else {
        let source = session
            .plan()
            .sources
            .iter()
            .find(|source| source.output_channel == channel)
            .map_or("unknown source", |source| source.id.as_str());
        format!(
            "timing chirps on output channel {channel} (source '{source}'); no fixed emitter declared, takes stay arrival-aligned and coherent use is forbidden"
        )
    }
}

#[cfg(test)]
mod tests;
