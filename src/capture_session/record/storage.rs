//! Common session evidence for in-memory test backends and native file capture.

// Rust guideline compliant 2026-02-21

#[cfg(feature = "capture-io")]
use crate::signal_recorder::multi_capture::{CapturedFile, FileCaptureResult};
use crate::signal_recorder::{MultiCaptureResult, write_wav_file};
use std::borrow::Cow;
use std::path::Path;

pub(super) trait SessionResult {
    fn into_session(self) -> Result<SessionCapture, String>;
}

pub(super) struct SessionCapture {
    pub input_device_ids: Vec<String>,
    pub output_device_id: String,
    pub sample_rate_hz: u32,
    pub recordings: Vec<RawRecording>,
    pub input_sample_formats: Vec<String>,
    pub output_sample_format: String,
}

enum RawData {
    Memory(Vec<f32>),
    #[cfg(feature = "capture-io")]
    File(CapturedFile),
}

pub(super) struct RawRecording {
    pub samples: usize,
    pub peak_amplitude: f32,
    pub clipped_samples: usize,
    data: Option<RawData>,
}

impl RawRecording {
    pub fn save(&mut self, path: &Path, rate: u32) -> Result<(), String> {
        match self
            .data
            .take()
            .expect("a raw take is published exactly once")
        {
            RawData::Memory(samples) => write_wav_file(path, &samples, rate, 1),
            #[cfg(feature = "capture-io")]
            RawData::File(file) => {
                if file.sample_rate_hz != rate {
                    return Err("capture WAV rate changed before publication".into());
                }
                file.persist(path)
            }
        }
    }

    pub fn wake_samples(&self, max_frames: usize) -> Result<Cow<'_, [f32]>, String> {
        if self.samples > max_frames {
            return Err("wake-up capture exceeds its read budget".into());
        }
        match self
            .data
            .as_ref()
            .expect("wake-up evidence has not been published")
        {
            RawData::Memory(samples) => Ok(Cow::Borrowed(samples)),
            #[cfg(feature = "capture-io")]
            RawData::File(file) => {
                let bytes = crate::artifacts::read_file(file.path(), max_frames * 4 + 4096)?;
                let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes))
                    .map_err(|error| error.to_string())?;
                reader
                    .samples::<f32>()
                    .collect::<Result<Vec<_>, _>>()
                    .map(Cow::Owned)
                    .map_err(|error| error.to_string())
            }
        }
    }
}

impl SessionResult for MultiCaptureResult {
    fn into_session(self) -> Result<SessionCapture, String> {
        let recordings = self
            .recordings
            .into_iter()
            .map(|samples| {
                if samples.iter().any(|sample| !sample.is_finite()) {
                    return Err("capture backend returned invalid samples".to_string());
                }
                Ok(RawRecording {
                    samples: samples.len(),
                    peak_amplitude: samples
                        .iter()
                        .fold(0.0_f32, |peak, sample| peak.max(sample.abs())),
                    clipped_samples: samples.iter().filter(|sample| sample.abs() >= 1.0).count(),
                    data: Some(RawData::Memory(samples)),
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(SessionCapture {
            input_device_ids: self.input_device_ids,
            output_device_id: self.output_device_id,
            sample_rate_hz: self.sample_rate_hz,
            recordings,
            input_sample_formats: self.input_sample_formats,
            output_sample_format: self.output_sample_format,
        })
    }
}

#[cfg(feature = "capture-io")]
impl SessionResult for FileCaptureResult {
    fn into_session(self) -> Result<SessionCapture, String> {
        let recordings = self
            .recordings
            .into_iter()
            .map(|file| {
                if file.sample_rate_hz != self.sample_rate_hz
                    || !file.peak_amplitude.is_finite()
                    || file.clipped_samples > file.samples
                {
                    return Err("file capture backend returned invalid evidence".to_string());
                }
                Ok(RawRecording {
                    samples: file.samples,
                    peak_amplitude: file.peak_amplitude,
                    clipped_samples: file.clipped_samples,
                    data: Some(RawData::File(file)),
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(SessionCapture {
            input_device_ids: self.input_device_ids,
            output_device_id: self.output_device_id,
            sample_rate_hz: self.sample_rate_hz,
            recordings,
            input_sample_formats: self.input_sample_formats,
            output_sample_format: self.output_sample_format,
        })
    }
}
