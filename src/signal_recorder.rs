//! Signal generation and recording module
//!
//! This module provides functionality to generate test signals, play them back,
//! record the output, and analyze the results.

mod build;
mod choose;
mod consts;
#[cfg(all(not(target_os = "ios"), any(test, feature = "test-util")))]
pub mod fake;
#[cfg(all(test, not(target_os = "ios")))]
mod fake_tests;
mod generate;
mod measurement;
mod misc;
#[cfg(not(target_os = "ios"))]
pub mod multi_capture;
#[cfg(not(target_os = "ios"))]
mod playback;
mod probe;
mod quality;
mod record;
#[cfg(not(target_os = "ios"))]
mod recorder;
mod recording_session;
mod run;
mod signal_params;
mod signal_type;
#[cfg(test)]
mod tests;
mod types;
mod write;

pub use consts::*;
#[cfg(all(not(target_os = "ios"), any(test, feature = "test-util")))]
pub use fake::{FakeAudioConfig, FakeAudioProfile, FakePlayback, FakeRecorder, VirtualWire};
pub use generate::*;
pub use misc::*;
#[cfg(not(target_os = "ios"))]
pub use playback::*;
pub use probe::*;
pub use quality::CaptureAnalysis;
pub use record::*;
#[cfg(not(target_os = "ios"))]
pub use recorder::{
    CpalRecorder, NegotiatedInput, RecordedTake, SweepRecorder, check_input_channels,
};
pub use recording_session::*;
pub use run::*;
pub use signal_params::*;
pub use signal_type::*;
pub use types::*;
pub use write::*;
