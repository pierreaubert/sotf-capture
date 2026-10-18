# Changelog

User-facing changes to `sotf-capture` are recorded here.

## 0.6.0 — 2026-10-18

### Changed

- Imported CTC and legacy-session artifacts must be single portable filenames.
  Symlinks, external paths, oversized inputs, and conflicting output destinations
  are rejected. CTC merging publishes a complete validated bundle atomically.
- CTC ears and electrical loopback must share one input device and timeline.
  All three recordings use one electrical clock mapping before cropping, preserving
  physical ear delays through startup offsets and drift. Missing loopback markers
  prevent CTC publication.
- Auxiliary playback retains one mono waveform and a routing schedule instead of
  allocating full-length buffers for every silent hardware channel.
- Capture storage has a 512 MiB aggregate budget. Per-device input rings store
  only selected channels instead of reserving a 33-float frame for every input.
- Sweep backends expose asynchronous health and per-clock input progress so
  frontends can preserve acquisition failures and independent input origins.

### Fixed

- Sweep and multi-device acquisition share readiness, timeout, cancellation, and
  cleanup handling. Every input clock must become ready and deliver its own tail;
  partial startup and stop failures release all streams.
- Native output completion includes the callback's reported presentation latency,
  rather than treating submission of the last output buffer as audible completion.
- Native sweep and auxiliary stream errors are latched for the control thread;
  callback error/overflow paths no longer invoke the logger. Lost samples fail
  acquisition rather than producing an accepted measurement.
- Auxiliary capture drains its bounded input ring while running, waits for real
  input readiness, and checks errors, cancellation, and monotonic deadlines.
- Probe, bass-anchor, and SPL calibration duration budgets are checked before waveform generation,
  including aggregate sequence duration and negotiated input storage.
- Wake-up bursts obey the requested capture amplitude ceiling.
- Quiet requested levels remain quiet throughout stimulus generation; the previous
  -40 dBFS floor no longer raises lower requested levels.
- Raw journals remain nonterminal until every declared auxiliary pass is committed;
  processing rejects complete journals missing declared auxiliary results.
- Clock fitting recognizes late positive marker lags wrapped by circular correlation.
- Localization bench calibration is frozen and hashed before acquisition, resolved
  consistently, and retained with raw loopback evidence in portable handoffs.
- Native sweep capture/playback use supported channel/rate/PCM-format tuples,
  including integer PCM. Measurement selectors require exact IDs or unique complete
  names; explicit ASIO selectors use the requested host or fail.
- Loaded metrics preserve unavailable bins as gaps (JSON null), reject non-finite
  frequency grids, and restore each missing metric independently.
- Ordinary sweep analysis borrows unchanged raw buffers; long emulated room tails
  use FFT convolution while preserving their complete decay.

### Tests

- Add adversarial file-import regressions and independent CTC startup/drift delay
  recovery coverage. A virtual clock exercises the production sweep lifecycle with
  delayed independent inputs, stalled output, asynchronous failures, cancellation,
  and stop errors. Prepared-stream tests cover partial-start cleanup and actual
  input-ring discontinuities. Long convolution is checked against a direct-sum oracle.

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
