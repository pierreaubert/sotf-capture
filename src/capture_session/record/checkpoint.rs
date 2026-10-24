//! Durable acquisition phase and committed-result identities.

// Rust guideline compliant 2026-02-21

use super::{RawCaptureManifest, RawCaptureStatus};
use serde::{Deserialize, Serialize};

/// Acquisition pass being attempted when the journal was last saved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CapturePhase {
    /// Freezing supporting files before opening streams.
    Preparing,
    /// Probing the audio chain before measurement passes.
    Waking,
    /// Recording the identified source and repeat on all declared inputs.
    Sweep {
        /// Source identity from the plan.
        source_id: String,
        /// Zero-based repeat index.
        repeat_index: u32,
    },
    /// Sweeps are committed; declared auxiliary passes remain.
    AwaitingAuxiliary,
    /// Acquiring the declared delay probe.
    Probe,
    /// Acquiring the declared bass anchor.
    BassAnchor,
    /// All declared acquisition passes have been committed.
    Complete,
}

/// One completed source/repeat pass on a shared loopback input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapturePassIdentity {
    /// Source identity from the plan.
    pub source_id: String,
    /// Zero-based repeat index.
    pub repeat_index: u32,
}

/// Active phase and explicit completion bitmap bound to committed journal results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureCheckpoint {
    /// Active pass, retained on failure/cancellation to identify the interrupted work.
    pub phase: CapturePhase,
    /// Stable IDs of committed microphone takes, in journal order.
    pub completed_take_ids: Vec<String>,
    /// Committed loopback passes, in journal order.
    pub completed_loopbacks: Vec<CapturePassIdentity>,
    /// Whether the declared probe result has been committed.
    pub probe_completed: bool,
    /// Whether the declared bass-anchor result has been committed.
    pub bass_anchor_completed: bool,
}

fn snapshot(manifest: &RawCaptureManifest, phase: CapturePhase) -> CaptureCheckpoint {
    CaptureCheckpoint {
        phase,
        completed_take_ids: manifest
            .takes
            .iter()
            .map(|take| take.take_id.clone())
            .collect(),
        completed_loopbacks: manifest
            .loopback_files
            .iter()
            .map(|take| CapturePassIdentity {
                source_id: take.source_id.clone(),
                repeat_index: take.repeat_index,
            })
            .collect(),
        probe_completed: manifest.probe_pass.is_some(),
        bass_anchor_completed: manifest.bass_anchor_pass.is_some(),
    }
}

#[cfg(any(test, feature = "capture-io"))]
pub(super) fn set_phase(manifest: &mut RawCaptureManifest, phase: CapturePhase) {
    manifest.checkpoint = Some(snapshot(manifest, phase));
}

#[cfg(any(test, feature = "capture-io"))]
pub(super) fn refresh(manifest: &mut RawCaptureManifest) {
    if let Some(checkpoint) = &manifest.checkpoint {
        manifest.checkpoint = Some(snapshot(manifest, checkpoint.phase.clone()));
    }
}

pub(crate) fn validate(manifest: &RawCaptureManifest) -> Result<(), String> {
    let Some(checkpoint) = &manifest.checkpoint else {
        return Ok(());
    };
    if checkpoint != &snapshot(manifest, checkpoint.phase.clone())
        || checkpoint.completed_take_ids.iter().any(String::is_empty)
    {
        return Err("acquisition checkpoint disagrees with committed results".into());
    }
    if (checkpoint.phase == CapturePhase::Complete)
        != (manifest.status == RawCaptureStatus::RawComplete)
    {
        return Err("acquisition checkpoint phase disagrees with terminal status".into());
    }
    let valid = match &checkpoint.phase {
        CapturePhase::Sweep {
            source_id,
            repeat_index,
        } => {
            *repeat_index < manifest.plan.repeat_count
                && manifest
                    .plan
                    .sources
                    .iter()
                    .any(|source| &source.id == source_id)
        }
        CapturePhase::Probe => manifest.plan.probe.is_some(),
        CapturePhase::BassAnchor => manifest.plan.bass_anchor.is_some(),
        CapturePhase::AwaitingAuxiliary => {
            manifest.plan.probe.is_some() || manifest.plan.bass_anchor.is_some()
        }
        _ => true,
    };
    if !valid {
        return Err("acquisition checkpoint names an undeclared pass".into());
    }
    Ok(())
}
