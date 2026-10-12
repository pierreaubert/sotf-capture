//! Structured evidence and policy context for acoustic capture quality.

// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

/// Purpose for which a capture quality verdict is being interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityPolicy {
    /// Advisory checks for a valid swept-sine analysis.
    SweepAnalysis,
    /// Advisory checks for a single sweep used for magnitude analysis.
    SingleTake,
    /// Stronger evidence required before coherent multi-channel use.
    CoherentCapture,
}

/// Metric represented by a structured quality finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityMetric {
    TotalHarmonicDistortion,
    BroadbandSnr,
    BandSnr,
    Clipping,
    TimingMarkers,
}

/// Result state for a quality check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityStatus {
    Pass,
    Warning,
    Unknown,
}

/// Unit for a measured quality value or threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityUnit {
    Percent,
    Decibels,
    Samples,
    Count,
}

/// How a metric was estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityMethod {
    /// Harmonic-only Farina estimate from separated second through fifth harmonic IRs.
    FarinaHarmonicImpulseResponse,
    BroadbandPowerRatio,
    NoiseSubtractedBandPowerRatio,
    FullScaleSampleCount,
    TimingMarkerFit,
}

/// Threshold comparison used by a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdComparison {
    AtMost,
    AtLeast,
    EqualTo,
}

/// A frequency interval associated with a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QualityBand {
    /// Lower edge in Hz, inclusive.
    pub low_hz: f64,
    /// Upper edge in Hz, exclusive.
    pub high_hz: f64,
}

/// Numeric criterion applied to a measured value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QualityThreshold {
    pub comparison: ThresholdComparison,
    pub value: f64,
}

/// Actionable, machine-readable evidence from one capture quality check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityFinding {
    pub policy: QualityPolicy,
    pub metric: QualityMetric,
    pub status: QualityStatus,
    /// Missing when the metric could not be measured; never substitute zero.
    pub measured: Option<f64>,
    pub unit: QualityUnit,
    pub band: Option<QualityBand>,
    pub method: QualityMethod,
    pub threshold: Option<QualityThreshold>,
    /// Why evidence is unknown, when `measured` is absent.
    pub unknown_reason: Option<String>,
    /// Suggested next step for resolving a warning or unknown result.
    pub recommendation: Option<String>,
    /// Compatibility text for existing report and UI consumers.
    pub message: String,
}

impl QualityFinding {
    /// Construct a finding whose measured value is unavailable.
    pub fn unknown(
        policy: QualityPolicy,
        metric: QualityMetric,
        unit: QualityUnit,
        method: QualityMethod,
        reason: impl Into<String>,
        recommendation: Option<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            policy,
            metric,
            status: QualityStatus::Unknown,
            measured: None,
            unit,
            band: None,
            method,
            threshold: None,
            unknown_reason: Some(reason.into()),
            recommendation,
            message: message.into(),
        }
    }

    /// Construct an unknown finding while retaining a known acceptance threshold.
    pub fn unknown_with_threshold(
        policy: QualityPolicy,
        metric: QualityMetric,
        unit: QualityUnit,
        method: QualityMethod,
        threshold: QualityThreshold,
        reason: impl Into<String>,
        recommendation: Option<String>,
        message: impl Into<String>,
    ) -> Self {
        let mut finding = Self::unknown(
            policy,
            metric,
            unit,
            method,
            reason,
            recommendation,
            message,
        );
        finding.threshold = Some(threshold);
        finding
    }
}

/// SNR floor for advisory single-sweep analysis, in dB.
pub const SINGLE_TAKE_SNR_THRESHOLD_DB: f64 = 10.0;
/// SNR floor for coherent capture acceptance, in dB.
pub const COHERENT_CAPTURE_SNR_THRESHOLD_DB: f64 = 30.0;
/// THD warning threshold for Farina sweep analysis, in percent.
pub const THD_WARNING_THRESHOLD_PERCENT: f64 = 10.0;
