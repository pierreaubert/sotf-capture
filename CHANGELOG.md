# Changelog

User-facing changes to `sotf-capture` are recorded here.

## 0.5.2 — 2026-10-13

### Added

- Hardware-free capture regression tests using emulated playback and recording backends.
- A Fireface 5.0.4 recording plan with four microphones, ear-pair CTC measurements,
  a surveyed timing reference, electrical loopback, and probe/bass-anchor passes.

### Changed

- `capture` now plays each requested output sequentially while recording every
  requested microphone simultaneously. Input and output channel lists are
  independent: three outputs and four microphones produce twelve measurements.
  Equal-length lists also capture every combination instead of paired channels.
- Sweep THD quality control evaluates 80 Hz–16 kHz, intersected with the requested
  sweep range. Stimulus generation and exported responses retain the full requested band.
- `plan-validate` checks every microphone calibration file before reporting success.
  Missing, unreadable, or non-file paths are reported together with microphone IDs,
  attempted paths, and filesystem errors. Relative paths resolve from the plan directory.
- Session drift detection uses start/end timing chirps. Acquisition handoffs retain
  per-take settings and effective stimulus duration.

### Fixed

- Acoustic sweeps use sweep-specific lag detection from `math-dsp`, improving signal
  lock for room recordings dominated by bass resonances. Accepted timing is reused
  throughout sweep analysis and repeat alignment.
- CLI sweep bounds reach ESS analysis, enabling sweep timing and Farina harmonic
  distortion analysis instead of the generic signal-analysis path.
- Session recording supports four microphones plus a separate electrical loopback
  input, including five-channel callback buffers.
- Native audio streams wider than 64 channels are accepted when selected routes are
  valid, supporting the Fireface UFX+ 94-channel CoreAudio stream at 48 kHz.
- Measurement quality evidence survives mocked capture workflows. Repeated captures
  without reliable silent-window evidence report broadband SNR as unknown.
