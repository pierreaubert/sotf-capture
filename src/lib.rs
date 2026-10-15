#![allow(clippy::collapsible_if)]

//! Acoustic measurement capture sessions and stimulus playback.
//!
//! This crate owns the recording side of the measurement workflow: capture
//! types and session helpers ([`recording_types`], [`recording_helpers`]),
//! the multi-capture session pipeline ([`capture_session`]), the
//! UI-agnostic wizard model ([`wizard`]), low-level signal generation and
//! recording ([`signal_recorder`]), and audio device lookup ([`devices`]).
//! `RoomEQ` optimization and reporting stay in `autoeq`; the DAW engine stays
//! in `sotf-daw`. Both depend on this crate instead of duplicating capture.

// Rust guideline compliant 2026-02-21

pub mod capture_session;
pub mod devices;
pub mod ios;
pub mod live_level;
pub mod quality;
pub mod rate_limit;
pub mod recording_helpers;
pub mod recording_types;
pub mod signal_recorder;
pub mod wizard;
