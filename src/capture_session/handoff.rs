//! Immutable acquisition inventory published with the canonical RoomEQ projection.

use super::clock::io::ClockProcessedManifest;
use super::record::RawCaptureStatus;
use autoeq::capture_handoff::{
    CAPTURE_HANDOFF_FILENAME, CaptureArtifactIdentity, CaptureArtifactRole, CaptureCompletion,
    CaptureHandoff, CaptureTakeIdentity,
};
use autoeq::roomeq::{MeasurementSource, RoomConfig, SpeakerConfig};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

fn sha256_hex(digest: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut hex = String::with_capacity(digest.len() * 2);
    for &byte in digest {
        write!(&mut hex, "{byte:02x}").expect("writing to a String cannot fail");
    }
    hex
}

fn file_identity(
    root: &Path,
    name: &str,
    role: CaptureArtifactRole,
) -> Result<CaptureArtifactIdentity, String> {
    if !autoeq::capture_handoff::portable_capture_filename(name) {
        return Err("capture handoff inventory needs portable local filenames".into());
    }
    let path = root.join(name);
    let metadata = std::fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
    if !metadata.file_type().is_file() {
        return Err("capture handoff inventory refuses nonregular files".into());
    }
    let mut input = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new();
    let mut count = 0_u64;
    let mut buffer = [0_u8; 65536];
    loop {
        let read = input.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        count += read as u64;
        hash.update(&buffer[..read]);
    }
    if count != metadata.len() {
        return Err("capture artifact changed while building its inventory".into());
    }
    Ok(CaptureArtifactIdentity {
        file: name.into(),
        role,
        bytes: count,
        sha256: sha256_hex(&hash.finalize()),
    })
}

fn projected_response_file(
    configuration: &RoomConfig,
    source_id: &str,
    microphone_id: &str,
) -> Result<String, String> {
    let Some(SpeakerConfig::Single(MeasurementSource::Multiple(source))) =
        configuration.speakers.get(source_id)
    else {
        return Err("canonical capture source is unavailable".into());
    };
    let capture = source
        .provenance
        .capture
        .as_ref()
        .ok_or("canonical capture provenance is unavailable")?;
    let index = capture
        .takes
        .iter()
        .position(|take| take.microphone_id == microphone_id)
        .ok_or("canonical microphone is unavailable")?;
    source.measurements[index]
        .path()
        .and_then(|path| path.to_str())
        .map(str::to_owned)
        .ok_or_else(|| "canonical response filename is unavailable".into())
}

/// Publish exact artifact identities after a complete canonical projection exists.
///
/// Raw audio snapshots are copied before analysis by the clock-processing stage.
/// The inventory is the final publication marker; it does not certify hardware
/// calibration, coherent eligibility, or listening outcomes.
///
/// # Errors
/// Rejects incomplete projections, missing artifacts, invalid paths, changed files,
/// unsupported status, inconsistent take identities, and publication failures.
pub fn publish_capture_handoff(
    report: &ClockProcessedManifest,
    status: RawCaptureStatus,
    raw_journal_bytes: &[u8],
    output: &Path,
) -> Result<CaptureHandoff, String> {
    let configuration_file = report
        .recording_manifest
        .clone()
        .ok_or("canonical recording projection is unavailable")?;
    let configuration = super::manifest::recording_configuration(report)?;
    let mut roles = BTreeMap::from([(
        configuration_file.clone(),
        CaptureArtifactRole::Configuration,
    )]);
    for name in [
        "capture-clock.json",
        "capture-raw.json",
        "stimulus.wav",
        "timing-chirp.wav",
    ] {
        roles.insert(name.into(), CaptureArtifactRole::SupportingEvidence);
    }
    let selected_take_ids = report.selected_take_ids.as_ref().map(|ids| {
        ids.iter()
            .map(String::as_str)
            .collect::<std::collections::HashSet<_>>()
    });
    let mut takes = Vec::new();
    for processed in &report.takes {
        let source_id = &processed.raw.source_id;
        let mic_id = &processed.raw.microphone_id;
        let is_selected = selected_take_ids
            .as_ref()
            .is_some_and(|ids| ids.contains(processed.raw.take_id.as_str()));
        let response_file = if is_selected {
            Some(projected_response_file(&configuration, source_id, mic_id)?)
        } else {
            processed.analysis.as_ref().and_then(|analysis| {
                analysis.magnitude_file.clone().or_else(|| {
                    analysis
                        .common_reference
                        .as_ref()
                        .map(|phase| phase.response_file.clone())
                })
            })
        };
        let complex = response_file.as_ref().is_some_and(|response_file| {
            processed
                .analysis
                .as_ref()
                .and_then(|analysis| analysis.common_reference.as_ref())
                .is_some_and(|phase| phase.response_file == *response_file)
        });
        let raw_audio_file = processed.raw.wav_file.clone();
        let calibration = report
            .calibrations
            .iter()
            .find(|calibration| calibration.microphone_id == *mic_id)
            .ok_or("calibration snapshot is unavailable")?;
        roles.insert(raw_audio_file.clone(), CaptureArtifactRole::RawAudio);
        roles.insert(
            processed.audio_file.clone(),
            CaptureArtifactRole::ProcessedAudio,
        );
        roles.insert(calibration.file.clone(), CaptureArtifactRole::Calibration);
        if let Some(response_file) = &response_file {
            roles.insert(
                response_file.clone(),
                if complex {
                    CaptureArtifactRole::ComplexResponse
                } else {
                    CaptureArtifactRole::MagnitudeResponse
                },
            );
        }
        if let Some(analysis) = &processed.analysis {
            if let Some(file) = &analysis.magnitude_file {
                roles
                    .entry(file.clone())
                    .or_insert(CaptureArtifactRole::SupportingEvidence);
            }
            if let Some(phase) = &analysis.common_reference {
                for file in [&phase.response_file, &phase.impulse_file] {
                    roles
                        .entry(file.clone())
                        .or_insert(CaptureArtifactRole::SupportingEvidence);
                }
            }
        }
        takes.push(CaptureTakeIdentity {
            take_id: processed.raw.take_id.clone(),
            source_id: source_id.clone(),
            repeat_index: processed.raw.repeat_index,
            raw_audio_file,
            processed_audio_file: processed.audio_file.clone(),
            response_file,
            calibration_file: calibration.file.clone(),
            provenance: super::manifest::take_provenance(report, processed)?,
        });
    }
    let parent_bytes = std::fs::read(output.join("capture-raw.json"))
        .map_err(|error| format!("cannot verify preserved parent inventory: {error}"))?;
    if parent_bytes != raw_journal_bytes {
        return Err(
            "preserved parent inventory differs from exact acquisition journal bytes".into(),
        );
    }
    // Bind producer-owned evidence only. Finder metadata and unrelated notes
    // are not acquisition artifacts and must not invalidate a moved bundle.
    let artifacts = roles
        .into_iter()
        .map(|(file, role)| file_identity(output, &file, role))
        .collect::<Result<Vec<_>, _>>()?;
    let handoff = CaptureHandoff {
        version: 1,
        producer: "sotf-capture".into(),
        producer_version: env!("CARGO_PKG_VERSION").into(),
        session_id: sha256_hex(&Sha256::digest(raw_journal_bytes)),
        completion: match status {
            RawCaptureStatus::RawComplete => CaptureCompletion::Complete,
            RawCaptureStatus::Cancelled => CaptureCompletion::Cancelled,
            RawCaptureStatus::Failed => CaptureCompletion::Failed,
            RawCaptureStatus::Capturing => {
                return Err("capture has no terminal acquisition status".into());
            }
        },
        sample_rate_hz: report.plan.sample_rate_hz,
        source_ids: report
            .plan
            .sources
            .iter()
            .map(|source| source.id.clone())
            .collect(),
        microphone_ids: report
            .plan
            .microphones
            .iter()
            .map(|mic| mic.id.clone())
            .collect(),
        repeat_count: report.plan.repeat_count,
        selected_take_ids: report.selected_take_ids.clone(),
        parent_inventory_file: Some("capture-raw.json".into()),
        configuration_file,
        artifacts,
        takes,
    };
    handoff.validate()?;
    let bytes = serde_json::to_vec_pretty(&handoff).map_err(|error| error.to_string())?;
    crate::recording_helpers::save_recording_session_json(
        &output.join(CAPTURE_HANDOFF_FILENAME),
        &bytes,
    )
    .map_err(|error| error.to_string())?;
    Ok(handoff)
}

#[cfg(test)]
mod hash_tests {
    use super::{Sha256, sha256_hex};
    use sha2::Digest;

    #[test]
    fn sha256_hex_matches_the_known_lowercase_digest() {
        let digest = Sha256::digest(b"abc");
        assert_eq!(
            sha256_hex(&digest),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
