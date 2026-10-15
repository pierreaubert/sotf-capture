//! Signal generation and recording module
//!
//! This module provides functionality to generate test signals, play them back,
//! record the output, and analyze the results.

mod build;
mod choose;
mod consts;
#[cfg(any(test, feature = "test-util"))]
pub mod fake;
#[cfg(test)]
mod fake_tests;
mod generate;
mod measurement;
mod misc;
pub mod multi_capture;
mod playback;
mod probe;
mod quality;
mod record;
mod recorder;
mod recording_session;
mod run;
mod signal_params;
mod signal_type;
#[cfg(test)]
mod tests;
mod types;
pub(crate) mod wake;
mod write;

pub use consts::*;
#[cfg(any(test, feature = "test-util"))]
pub use fake::{FakeAudioConfig, FakeAudioProfile, FakePlayback, FakeRecorder, VirtualWire};
pub use generate::*;
pub use misc::*;
pub use playback::*;
pub use probe::*;
pub use quality::CaptureAnalysis;
pub use record::*;
pub use recorder::{
    CpalRecorder, NegotiatedInput, RecordedTake, SweepRecorder, check_input_channels,
};
pub use recording_session::*;
pub use run::*;
pub use signal_params::*;
pub use signal_type::*;
pub use types::*;
pub use write::*;
