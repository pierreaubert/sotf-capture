//! Signal generation and recording module
//!
//! This module provides functionality to generate test signals, play them back,
//! record the output, and analyze the results.

mod build;
#[cfg(feature = "capture-io")]
mod choose;
mod consts;
#[cfg(any(test, feature = "test-util"))]
pub mod fake;
#[cfg(test)]
mod fake_tests;
mod generate;
pub mod level_meter;
mod measurement;
mod misc;
#[cfg(feature = "capture-io")]
pub mod multi_capture;
mod playback;
mod probe;
mod quality;
mod record;
mod recorder;
mod recording_session;
#[cfg(feature = "capture-io")]
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
pub use level_meter::{CaptureLevelMeter, CaptureLevelSnapshot, GainAdvice, LevelZone};
pub use misc::*;
pub use playback::*;
#[cfg(feature = "capture-io")]
pub use probe::*;
pub use quality::CaptureAnalysis;
pub use record::*;
#[cfg(feature = "capture-io")]
pub use recorder::CpalRecorder;
pub use recorder::{NegotiatedInput, RecordedTake, SweepRecorder, check_input_channels};
pub use recording_session::*;
#[cfg(feature = "capture-io")]
pub use run::*;
pub use signal_params::*;
pub use signal_type::*;
pub use types::*;
pub use write::*;
