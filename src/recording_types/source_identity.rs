use serde::{Deserialize, Serialize};

/// Address of one source in the imported speaker/driver hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecordingSourceIdentity {
    pub speaker: String,
    pub driver_index: Option<usize>,
    pub measurement_index: usize,
}
