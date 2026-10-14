# Plan: automatic speaker–microphone distances and 3D localization from chirps

Status: draft for review — no code yet. This document is the contract the
implementation must satisfy; where it names a threshold or a refusal, the
code must fail closed (report `None` plus a reason) rather than fabricate a
number, matching the existing clock and reflection stages.

Scope assumption (fixed for v1, decided 2026-10-14): **all microphones are
the same brand/type and are all connected to the same audio interface.**
Section 3 explains what this buys and what it removes from the design.
Multi-device rigs are an explicit non-goal for v1 (§2.2, §14).

Owners: capture = this crate (`sotf-capture`); DSP primitives =
`math-audio/crates/math-dsp`; RoomEQ consumption = `autoeq` (later phase).

---

## 1. Summary and decision

`sotf-capture` already records everything localization needs: for every
source take, one stimulus (2–8 kHz timing chirps bracketing a log sweep)
is played on the source channel with the chirps optionally rerouted to a
fixed emitter channel, while all microphones record simultaneously
(`src/capture_session/record.rs:598-620`,
`src/capture_session/protocol.rs:51-106`). The clock stage already turns
chirp delays into a per-take affine clock fit
(`src/capture_session/clock.rs:95-262`).

Decision: build localization as a **new offline processing stage** in
`plan-process`, after clock correction and before/alongside the existing
reflection analysis, in three layers shipped in order:

1. **Range matrix** — one absolute distance per source×microphone pair,
   with an uncertainty and a quality gate per cell.
2. **3D speaker positions** — one multilaterated position per source from
   the surveyed microphone positions, with covariance and residuals.
3. **RoomEQ handoff** — publish ranges/positions as hash-bound evidence
   next to the existing provenance; never silently overwrite declared
   geometry.

The microphone-localization inverse (surveyed speakers, unknown mics)
and full joint self-calibration are designed here but scheduled last
(§11, Phase 4) and may be cut without invalidating Phases 1–3.

## 2. Goals and non-goals

### 2.1 Goals

- G1. For a completed `plan-record` session, automatically produce the
  full source×microphone distance matrix in metres, without any new
  user measurement, whenever the timing evidence supports it.
- G2. When at least four microphones have surveyed, non-coplanar
  positions, produce a 3D position estimate per source in the plan's
  coordinate frame, with per-axis uncertainty, RMS residual, and an
  explicit mirror/degeneracy flag.
- G3. Every published number carries its evidence: arrival time, clock
  basis, sound speed used, uncertainty budget, and the take IDs it came
  from. A reader of `capture-clock.json` alone can reproduce the range.
- G4. Refuse, with a named reason, whenever the evidence cannot support
  the claim (weak chirp, reflection picked instead of direct sound,
  degenerate geometry, unknown latency). Refusal is per source or per
  cell, never a whole-run abort; the rest of the session still processes.
- G5. No change to acquisition: no new stimulus, no new plan requirement
  for existing users. Localization consumes the takes that already exist.

### 2.2 Non-goals (v1)

- N1. Microphones on separate interfaces/USB devices with independent
  clocks (e.g. several UMIKs each on its own clock). The single-interface
  assumption (§3) removes per-device offsets; supporting N1 later means
  re-adding a per-device offset unknown (§14) and must not be smuggled
  into v1 by assuming the current clock fit generalizes.
- N2. Moving-microphone (MMM) captures. A moving mic has no fixed
  position to solve for; ranges during motion are out of scope.
- N3. Estimating room dimensions, reflection positions, or a full room
  model. The existing reflection/DOA stage
  (`src/capture_session/reflections.rs`) stays as is; localization only
  adds the direct-sound range/position layer beneath it.
- N4. Replacing surveyed microphone positions in the plan. Estimated
  speaker positions are *evidence beside* the plan, not an edit of it;
  feeding them back into a plan is a separate, explicit user action.
- N5. Sub-sample bragging rights. The deliverable is centimetre-class
  positions in real rooms with an honest uncertainty, not a laboratory
  millimetre claim (§6.3).

## 3. The single-interface, matched-microphone assumption

### 3.1 What "same interface" means here

- All microphones land on input channels of one interface, sampled by
  one ADC clock. Inter-channel skew on such hardware is a fixed,
  sub-sample converter/multiplexer constant, identical for every take —
  it is part of the one-time latency calibration (§5.4), not a per-take
  unknown.
- Playback is assumed to leave through the same interface in the
  reference rig (the `CaptureSessionPlan.output_device` is one device
  and the microphones share one input device string pattern in the
  examples, e.g. `examples/home-theater-5.1-sub.json`). If DAC and ADC
  are in fact two boxes, §5.3 Mode B applies and absolute ranging needs
  the loopback or the fixed emitter; the code must detect which case it
  is in from clock evidence, not from device-name strings.
- Consequence for the clock model: there is **one** input clock. All
  takes share one skew (up to take-to-take wander), and there is exactly
  **one** unknown clock offset for the whole session instead of one per
  microphone. The existing per-take fit in `correct_capture_clock`
  remains correct (it fits a superset of this model), and localization
  additionally *pools* the per-take fits: a session-level skew is the
  median of per-take skews, and a take whose skew disagrees with the
  pool beyond the existing 100 ppm wander warning
  (`clock/io.rs:186-195`, `CLOCK_SKEW_SPREAD_PPM`) is excluded from
  localization with reason `clock_outlier`.

### 3.2 What "same brand/type" means here

- Matched capsules have, to first order, the same frequency response,
  the same group delay versus frequency, and the same acoustic-centre
  offset. In the observation model (§4) each microphone contributes a
  delay bias `b_mic`; matched mics make `b_mic ≈ b_common`, a single
  constant shared by every channel.
- `b_common` cancels exactly in every inter-microphone difference
  (TDOA, DOA, position *shape*) and shifts every absolute range by the
  same `c · b_common`. One bench calibration (mic against a reference
  at a taped distance, §13.4) measures `b_common` once per mic *model*,
  not once per session.
- It does **not** remove per-unit sensitivity differences: per-microphone
  calibration files stay mandatory and unchanged
  (`CaptureMicrophone.calibration_file`,
  `src/capture_session.rs:92-93`). Matched type is a delay statement,
  not a level statement.
- The 2–8 kHz chirp band sits where magnitude-only capsule calibration
  is typical; matched group delay is precisely what makes a single
  chirp-derived delay comparable across channels in that band. This is
  the same assumption the DOA band limits already rely on
  (`math-dsp/src/capture_array.rs:12-18`).

### 3.3 What the assumption removes from v1

| Removed problem | Why it is gone | Where it returns |
| --- | --- | --- |
| Per-mic clock offset/skew unknowns | One ADC clock (§3.1) | §14 multi-device extension |
| Per-mic delay-bias unknowns | Matched group delay (§3.2) | Per-unit phase calibration, future |
| Cross-device take alignment | All mics in one simultaneous capture per source (`record.rs:598+`) | Never for this plan shape |
| Per-mic resampling before comparison | One clock → at most one common resample | §14 |

What it does **not** remove: DAC/ADC absolute latency, loudspeaker-side
electronic/acoustic delay, sound-speed error, and multipath. Those are
the real design problems, addressed in §5–§6.

## 4. Observation model

### 4.1 Notation

- Sources `s = 1..S` (plan `sources`), microphones `m = 1..M`
  (plan `microphones`, `M ∈ 2..=4` today, `src/capture_session.rs:352`).
- Microphone positions `p_m ∈ R³` are surveyed inputs
  (`CaptureMicrophone.position_m`, metres, one session frame).
- Source positions `x_s ∈ R³` are the unknowns of Goal G2. Today
  `CaptureSource` is only `{ id, output_channel }`
  (`src/capture_session.rs:107-112`); §8 adds an *optional* surveyed
  position for the inverse problem and for validation comparisons.
- `c` = speed of sound used for the session, from
  `CaptureTimingReference.sound_speed_m_s` when a reference is declared,
  else a new optional plan field, else 343.0 m/s with an uncertainty
  large enough to show in the budget (§6.2). Never a silent constant.
- `Fs` = `plan.sample_rate_hz`.

### 4.2 One take, one equation

For the take of source `s` recorded at microphone `m`, define:

```
tau[s,m]  = measured arrival of the direct sound, in stimulus-clock
            samples, after clock correction (§7.2)
d[s,m]    = c · (tau[s,m] − tau_emit − L) / Fs          (range, metres)
```

where `tau_emit` is the stimulus-clock sample at which the relevant
signal left the DAC (known exactly from `CaptureStimulusLayout`) and
`L` is the total non-acoustic latency in samples (DAC + interface +
ADC + any fixed converter skew), measured electrically per §5.4. Every
term except `tau[s,m]` is session-level, not per-cell: with one
interface there is one `L`, one clock, one `c`. That is the whole
reason the single-interface assumption matters — the range matrix has
`S·M` measurements and only a handful of shared nuisance parameters.

The loudspeaker contributes its own delay `g_s` (DSP/crossover group
delay at the measurement band plus acoustic-centre offset):
`d_measured = d_geometric + c·g_s`. `g_s` is indistinguishable from
distance within one source's row, so v1 reports ranges and positions as
**acoustic-centre positions at the chirp band**, states this verbatim
in the output (§8.3), and accepts an optional per-source delay
calibration (§5.5) that subtracts a measured `g_s`. It never pretends
the chirp found the cabinet front baffle.

### 4.3 Two signals, two arrival estimators

The stimulus carries two usable signals, and they answer different
questions (routing in `record.rs:604-620`):

- **Timing chirps ride the emitter channel** (`timing_reference` channel
  when declared, else the first source's channel). Their arrival at
  every mic, fitted by `estimate_chirp_tdoa`
  (`math-dsp/src/capture_tdoa.rs:91-229`, 2–8 kHz, fractional-sample,
  `confidence_db ≥ 6 dB` validity), establishes the clock and, when the
  emitter position is surveyed, each mic's absolute time origin. The
  chirp says nothing about a *source* position unless the emitter *is*
  that source.
- **The sweep rides the source channel.** Its deconvolved impulse
  response (the `write_phase` path, `src/capture_session/phase.rs`,
  origin `shared_stimulus_sweep_start`, delay preserved — pinned today
  by the test `exported_ir_retains_delay_and_calibration_retains_phase`)
  contains the source's direct arrival. The source range comes from the
  sweep IR's direct-sound onset/peak (the same direct-sound detection
  `math-rir` already performs for the reflection stage,
  `toa_sample` / `direct_sound()`), cross-checked against a chirp-band
  matched-filter arrival measured on the sweep's own 2–8 kHz content.

Agreement rule (quality gate Q3, §9): if the IR-peak arrival and the
band-limited matched-filter arrival differ by more than **1.0 sample**,
the cell is flagged `arrival_ambiguous` and excluded from positioning
(it may still appear in the range matrix, marked). This is the primary
defence against "strongest reflection picked instead of direct sound":
a reflection can win one estimator, rarely both at the same sample.

Subwoofer caveat: a sub reproduces little 2–8 kHz content, so the
chirp-band cross-check is invalid for sub sources. For sources whose
sweep energy above 300 Hz is negligible (measured, not declared), the
arrival estimator switches to the low band (sweep deconvolution onset
in 20–300 Hz plus the existing bass-anchor phase evidence,
`CaptureBassAnchor`) and the cell uncertainty is widened accordingly.
The plan must state which band produced each cell (§8.3 `band` field);
mixing bands inside one position solve is allowed only through the
per-cell uncertainties as weights, never silently.

## 5. Clock, latency, and time origin

### 5.1 What the existing clock stage already proves

`correct_capture_clock` (`clock.rs:95-262`) fits, per take: start/end
chirp lags (`estimate_chirp_tdoa`), skew (`estimate_clock_skew`, valid
within `MAX_PLAUSIBLE_SKEW_PPM = 5000 ppm`), a Cramér–Rao-style marker
uncertainty (`post_correction_uncertainty_us`), and a basis:

- `FixedAcousticReference` — a surveyed emitter's known distance was
  subtracted (`clock.rs:210-248`): corrected samples sit on the stimulus
  clock with acoustic travel preserved. **Absolute ranging eligible.**
- `ArrivalAligned` — arrivals aligned, emitter geometry unknown
  (`clock.rs:205-209`). Acoustic travel and clock offset are confounded.
  **Relative (TDOA) use only.**
- `Uncorrected` — no valid fit. **No timing use.**

Localization must consume this basis verbatim and never upgrade it.
The `COHERENT_TIMING_LIMIT_US = 50 µs` gate (`clock.rs:23`) stays a
clock-quality statement; localization adds its own, range-specific
gates in §9 and does not reuse 50 µs as a position criterion.

### 5.2 Refactor required: expose the fit, not only the resample

Today the fit's intermediate values (per-chirp lags, confidences) live
inside `correct_capture_clock` and only the resampled audio plus
provenance escape. Localization needs the same numbers without
re-running correlations on resampled audio (resampling before arrival
estimation would blur the very peak being measured, and re-running on
raw audio would duplicate the search-window logic).

Required change (Phase 1, §11): split `correct_capture_clock` into

```
pub(crate) struct CaptureClockFit {          // new, internal
    pub first:  TdoaEstimate,   // start chirp, stimulus-referenced
    pub second: TdoaEstimate,   // end chirp, stimulus-referenced
    pub skew:   ClockSkew,      // offset at stimulus zero, skew ppm
    pub marker_uncertainty_us: Option<f64>,
    pub basis:  CaptureClockBasis,
    ...
}
pub(crate) fn fit_capture_clock(manifest, take, recorded, reference)
    -> CaptureClockFit
pub fn correct_capture_clock(...)  // unchanged signature; calls
                                   // fit_capture_clock + resample
```

External behaviour of `correct_capture_clock` must be bit-identical
(same provenance values, same resampled output); the existing clock
tests (`src/capture_session/clock/tests.rs`, golden fixture
`tests/fixtures/capture-clock-golden.json`) are the regression net and
must pass unmodified. Per the repo's Rust rules, load the ms-rust
guidelines (universal + documentation + library) before touching this
file, keep `// Rust guideline compliant 2026-02-21`, and keep the
module docs in the canonical format.

### 5.3 Latency cases (detect, don't assume)

`L` (§4.2) is one number per session only if DAC and ADC share a clock
and a driver latency. Three detectable cases:

- **Mode A — one interface in+out.** DAC and ADC share the device
  clock; measured skew ≈ 0 (within a few ppm) and a loopback take (the
  plan `loopback` route, already recorded for CTC/bass-anchor) gives
  `L` directly: correlate the electrical loopback against the known
  stimulus; that lag *is* `L` plus a sub-sample converter constant.
  Absolute ranging is then available **without** a surveyed emitter.
- **Mode B — separate DAC and ADC, no loopback.** Skew is whatever the
  two crystals do; `L` is unknown. Absolute ranging requires the
  surveyed fixed emitter (the existing `FixedAcousticReference` path):
  the emitter's known distance to each mic converts chirp arrivals
  into per-mic time origins. This is exactly the geometry the code at
  `clock.rs:219-235` already computes — inverted.
- **Mode C — neither.** Only inter-mic differences exist. Publish TDOA
  (range differences per source, mic 1 as reference, mirroring
  `pairwise_tdoas_vs_first` in `math-dsp/src/capture_array.rs:416-435`),
  mark every absolute field `None`, reason
  `no_absolute_time_origin`. Speaker *shape* (relative geometry up to
  the §6.4 ambiguities) may still be solvable in Phase 4; v1 stops at
  TDOA here.

Mode is decided per session from evidence, in this priority: loopback
present and valid → A; else valid `FixedAcousticReference` on all used
takes → B; else C. The chosen mode and its evidence are stored in the
report (§8.3). A session must not mix modes across sources.

### 5.4 One-time bench calibration (per interface + mic model)

Deliverable of Phase 0 (§11), producing two constants stored in a small
versioned JSON the plan can reference:

1. `L` reference value: loopback lag of the bare interface (cable
   DAC→ADC), in samples at each supported rate. This value is operator
   reference only — absolute session `L` is unobservable from
   single-clock data (every clock fit's offset absorbs the electrical
   latency together with the capture-start offset, which `clock.rs`
   deliberately leaves arbitrary; the loopback-minus-fit residual is
   zero by construction, not a measurement). There is therefore no
   session-side bench comparison and no `latency_implausible` flag;
   the emitter anchor (Q6) is the session's electrical sanity check,
   and the three-repeat bench stability procedure below is the control
   for driver latency drift.
2. `b_common`: one mic of the session's model at a taped 1.000 m from a
   small full-range source, loopback-referenced; `b_common = measured
   range − 1.000 m`. Expected magnitude: a few millimetres to ~2 cm
   (capsule acoustic centre + fixed group delay). Applied as a single
   subtraction from every range. If never measured, `b_common = 0` and
   the report says so — positions remain internally consistent, with a
   common radial bias disclosed, not hidden.

### 5.5 Per-source electronic delay (optional input)

Amplifier/DSP latency (common in active speakers and AVRs: 0–30 ms)
is indistinguishable from metres of air. The plan gains an optional
`source_delay_calibration_ms` per source (default absent = 0, stated).
When a user has measured it (loopback through the speaker's DSP at
1 m, or manufacturer figure), ranges subtract it. The output always
reports both the raw acoustic-centre range and, when calibrated, the
corrected range. Never fold an assumed AVR delay into defaults.

## 6. Geometry: from ranges to 3D positions

### 6.1 Solver

Per source `s`, given ranges `d_m` to microphones at `p_m` with
per-range standard deviations `sigma_m` (§6.3):

```
minimize over x:  sum_m  w_m · rho( |x − p_m| − d_m ),  w_m = 1/sigma_m²
```

- `rho` = Huber loss (δ = 3σ) via iteratively reweighted least squares,
  so one reflection-corrupted range degrades weight instead of dragging
  the position. A range whose final residual exceeds 3.5σ is excluded
  and the solve repeated once; the exclusion is reported per cell.
- Optimization: Gauss–Newton with analytic Jacobian (rows are unit
  vectors `(x − p_m)/|x − p_m|`), Levenberg damping `1e-9`, max 100
  iterations, convergence at step < 1e-9 m. Deterministic: fixed
  iteration order over plan microphone order, no randomness anywhere.
- Initialization (deterministic, no user seed): (a) linear estimate by
  squared-range differencing against mic 0 (subtracting the mic-0
  sphere equation linearizes the system), solved by least squares;
  (b) if that fails or lands outside the room hull, fall back to the
  uncertainty-weighted centroid of microphones pulled outward along
  the mic-0→centroid axis by `d_0`. Both candidates are refined; the
  lower final cost wins. This mirrors the closed-form-then-refine
  pattern already used for DOA (`estimate_doa_ls`,
  `math-dsp/src/capture_array.rs:323-369`).
- Pure function, no I/O, no audio: `solve_position(mics, ranges) ->
  Option<PositionEstimate>` in the DSP layer (§8.2), so it is testable
  against synthetic geometry without a capture session.

### 6.2 Sound speed

`c` enters every range linearly: a fractional error in `c` is the same
fractional error in every distance (0.1 % ≈ 3 mm at 3 m). Sources, in
priority order: declared `timing_reference.sound_speed_m_s` (validated
today to 250–400 m/s with an uncertainty,
`src/capture_session.rs:513-527`); else a new optional plan field
`sound_speed { m_s, uncertainty_m_s, air_temperature_c? }`; else
343.0 ± 3.0 m/s (the ±3 covers roughly ±5 °C around 20 °C via
`c ≈ 331.3 + 0.606·T` and is deliberately honest). The uncertainty term
propagates into `sigma_m` exactly as the existing geometry bound does
(`clock.rs:230-235`: range × σc / c²), so a lazy `c` visibly widens the
position covariance instead of silently biasing it. Optional future:
fit `c` as an extra unknown when at least one surveyed inter-point
distance exists in the solve (e.g. the emitter), §14.

### 6.3 Uncertainty budget (per range, then per position)

| Term | Typical size at 48 kHz | Where it comes from |
| --- | --- | --- |
| Sample quantization after interpolation | ~0.05–0.2 sample (0.4–1.4 mm) | `estimate_chirp_tdoa` parabolic peak on the analytic correlation; synthetic tests pin < 0.05 sample (`capture_tdoa.rs` tests) |
| Chirp/IR arrival noise (CRLB-style) | `post_correction_uncertainty_us` model, `3σ + 0.5` sample floor | Reuse the §C4 model (`math-dsp/src/capture_tdoa.rs:306-354`), evaluated per cell from its own confidence |
| Clock residual after correction | included above via marker uncertainty; skew pooled (§3.1) | `CaptureClockProvenance.marker_uncertainty_us` |
| Mic survey error | plan `position_uncertainty_mm` per mic (1 mm compact, 5 mm spread) | Already declared; enters as position covariance, §6.1 weighting |
| Sound speed | `d · σc / c` (≈3 mm/m per °C unmeasured) | §6.2 |
| Matched-mic bias residual | after `b_common`, budget 2 mm | §5.4 bench value; stated if uncalibrated |
| Loudspeaker group-delay variation | unbounded in principle; gated, not budgeted | Q3 band-agreement gate (§4.3); excluded cells carry the reason |

Position covariance: `Cov(x) = (Jᵀ W J)⁻¹` at the solution (with
mic-position covariance folded into `W`), reported as per-axis σ in
metres and as GDOP = `sqrt(trace(Cov))/sigma_typical`. Positions are
published with σ, never as bare points. Expected honest outcome in a
normal room: 1–4 cm per axis; the §13 hardware test measures the real
figure instead of trusting this table.

### 6.4 Identifiability (checked before solving, not after failing)

| Microphones (surveyed) | Ranges per source | Result |
| --- | --- | --- |
| ≥ 4, non-coplanar, non-collinear | ≥ 4 valid | Unique 3D position (up to noise) |
| 3, non-collinear | 3 valid | Two mirror solutions across the mic plane — publish none, flag `mirror_ambiguous` with the position refused (same honesty rule as `reflections.rs:221-226`) |
| ≥ 4 but coplanar (flat seat row at one height) | ≥ 4 | Mirror ambiguity across the mic plane persists — elevation sign unobservable. Publish with `mirror_ambiguous: true`; recommend adding one mic at a different height |
| 2 | 2 | Circle of solutions — no position; ranges/TDOA only |
| Any, all mics collinear | any | Refuse positioning, reason `degenerate_geometry` |

Rank/planarity is tested on the surveyed `position_m` set directly
(SVD of centred positions; smallest singular value < 1 mm ⇒ coplanar,
mirroring `is_coplanar` in `math-dsp/src/capture_array.rs:89-96` and
the rank logic in `solve_direction`). These checks run on plan
geometry at `plan-validate` time as warnings (§10), so the user learns
*before* recording that their flat four-seat row cannot yield heights.

## 7. DSP pipeline (per `plan-process` run)

### 7.1 Where it hooks in

In `process_capture_session_with_options`
(`src/capture_session/clock/io.rs:331+`), after the per-take loop that
already computes `correct_capture_clock` + `analyze_capture` and after
`reflection_reports` are built (`clock/io.rs:613-617`). Localization
reads only: the plan, the stimulus layout, the chirp reference, each
take's raw WAV (already copied into the processed directory), each
take's `CaptureClockProvenance`, and the phase IRs when present. It
writes one new field on `ClockProcessedManifest` (§8.3) and one new
`pending_processing` transition (`"localization"`), and prints summary
lines from `plan_process` in `src/bin/sotf_capture.rs:871-934` next to
the existing clock lines. No raw-directory mutation, no device access.

### 7.2 Steps

1. **Session clock pool.** Collect per-take fits (§5.2). Compute pooled
   skew (median) and per-take offset on the stimulus clock. Mark
   `clock_outlier` takes (§3.1). If fewer than one valid fit per needed
   cell remains, that cell is `None` with the clock reason propagated.
2. **Mode decision** (§5.3): A / B / C from loopback validity and clock
   bases. Record mode + evidence.
3. **Electrical latency `L`** (Modes A/B): cancelled differentially,
   never measured absolutely. Microphone-minus-loopback arrivals share
   one capture call, so the shared clock offset, skew mapping, and `L`
   all cancel without resampling; in Mode B the emitter geometry
   identity plays the same role. Absolute session `L` is unobservable
   single-clock (see §5.4), so no bench comparison runs at process
   time; the bench lag map stays operator reference only.
4. **Emitter ranges (sanity anchor).** In Mode B, compute each mic's
   *measured* emitter range from chirp arrivals and compare with the
   surveyed emitter distance: residual > 1 sample on any mic ⇒ the
   survey or the mode is wrong; flag the session `emitter_mismatch`
   and fall back to Mode C rather than publishing biased absolutes.
   (This is the existing `clock.rs:219-225` distance, finally checked
   instead of assumed.)
5. **Source arrivals.** Per source×mic cell, on clock-corrected audio:
   (a) IR direct-sound arrival from the take's phase IR when present;
   (b) chirp-band matched-filter arrival on the sweep window.
   Apply gate Q3 (§4.3). Sub sources use the low-band variant (§4.3).
   Multi-repeat sessions: median arrival across repeats, with the
   inter-repeat spread added to `sigma` (repeats are never averaged
   into fake samples; the existing no-averaging rule stands,
   `src/capture_session.rs:270-271`).
6. **Ranges.** `d = c·(tau − tau_emit − L)/Fs − b_common − c·g_s`
   (last two terms optional calibrations, §5.4–5.5), each with the
   §6.3 sigma. Negative or > room-diagonal + 5 m ranges are refused
   per cell (`range_implausible`), never clamped.
7. **Positions.** §6.1 solve per source over valid cells. Publish §6.4
   flags. Sources failing identifiability keep their range row and get
   `position: null` plus the reason.
8. **Self-checks before publishing** (§9): leave-one-out range
   residuals, emitter anchor (step 4), and, when the plan declares a
   surveyed source position (§8.1), the estimated-vs-declared distance
   as a session quality metric (not a gate on publishing the estimate).

### 7.3 Determinism and bounds

Same inputs ⇒ same outputs, as for the clock stage: bounded reads
(existing 16 M-sample and 128 MB artifact limits stay), no network, no
threads with nondeterministic reduction order, `f64` throughout the
geometry path (positions are already `f64` in the plan). Arrival
estimation reuses `TdoaConfig::default()` band constants
(2–8 kHz) unless the sub-band variant applies; constants live next to
the existing ones in `math-dsp`, not as magic numbers in this crate.

## 8. Data model

### 8.1 Plan (`CaptureSessionPlan`) — additive, optional, serde-defaulted

```rust
// CaptureSource gains (old plans must still parse — deny_unknown_fields
// is on, so these are new *optional* keys, never renamed old ones):
#[serde(default, skip_serializing_if = "Option::is_none")]
pub position_m: Option<[f64; 3]>,              // surveyed, for inverse
                                               // mode + validation
#[serde(default, skip_serializing_if = "Option::is_none")]
pub position_uncertainty_mm: Option<f64>,
#[serde(default, skip_serializing_if = "Option::is_none")]
pub source_delay_calibration_ms: Option<f64>,  // §5.5

// Plan level:
#[serde(default, skip_serializing_if = "Option::is_none")]
pub sound_speed: Option<CaptureSoundSpeed>,    // §6.2 (ignored when a
                                               // timing_reference
                                               // already declares it)
#[serde(default, skip_serializing_if = "Option::is_none")]
pub localization_calibration_file: Option<PathBuf>, // §5.4 bench file
```

Validation rules (fail at `plan-validate`, not at solve time): finite
positions, uncertainty ≥ 0, delay calibration in [0, 100] ms, sound
speed in the existing 250–400 m/s window. Existing fixtures
(`tests/fixtures/capture-session.json`) must parse unchanged.
Microphone `position_m` stays **required** in v1 — making it optional
for the inverse problem is Phase 4 and needs its own migration care,
since today every consumer assumes it exists.

### 8.2 DSP layer (`math-dsp`, new module `capture_localization`)

Pure functions over slices and `[f64; 3]` arrays, in the style of
`capture_array.rs`/`capture_tdoa.rs`, re-exported from `lib.rs`:

- `solve_position(mics: &[[f64;3]], ranges: &[Option<(f64 /*d*/, f64 /*sigma*/)>]) -> Option<PositionEstimate>` — §6.1; `PositionEstimate { position_m, sigma_m: [f64;3], rms_residual_m, gdop, mirror_ambiguous, used: Vec<bool> }`.
- Geometry classifier: `geometry_rank(positions) -> GeometryRank { rank, coplanar, collinear }` via the centred-scatter eigenstructure (`SymmetricEigen`; singular value ≤ 1 mm counts as collapsed).
- Delay-to-distance conversion and its uncertainty combinator stay in `sotf-capture` (`propagation_to_meters`, `range_sigma_m`): they are coupled to session semantics (per-source delay calibration, sound-speed provenance, Q-gates), not reusable DSP. The stated-model style still applies (model in the doc comment, worst-case not best-case, `INFINITY` never `0` for unknown).

No `sotf-capture` types cross into `math-dsp`; the session crate adapts
plan/manifest types at the boundary, as `reflections.rs` already does
with `MicArray` (including its sound-speed scaling trick at
`reflections.rs:164-174` — localization must *not* copy that trick; it
uses the session `c` explicitly, since ranges, unlike DOA directions,
scale with `c`).

### 8.3 Processed output (`ClockProcessedManifest` gains one field)

```rust
#[serde(default)]
pub localization_report: Option<LocalizationReport>,

pub struct LocalizationReport {
    pub mode: LocalizationMode,        // LoopbackReferenced |
                                       // EmitterReferenced | RelativeOnly
    pub sound_speed_m_s: f64,
    pub sound_speed_uncertainty_m_s: f64,
    pub latency_samples: Option<f64>,  // L; None in RelativeOnly
    pub mic_bias_correction_m: f64,    // b_common applied (0 if none)
    pub ranges: Vec<SourceRanges>,     // one per plan source, plan order
    pub sources: Vec<SourcePosition>,  // one per plan source, plan order
    pub issues: Vec<String>,           // session-level, human-readable
}

pub struct SourceRanges {
    pub source_id: String,
    pub cells: Vec<RangeCell>,         // one per plan mic, plan order
}
pub struct RangeCell {
    pub microphone_id: String,
    pub distance_m: Option<f64>,       // None ⇔ refused, see reason
    pub sigma_m: Option<f64>,
    pub arrival_samples: Option<f64>,  // stimulus-clock, corrected
    pub band: Option<String>,          // "chirp_2_8k" | "low_20_300"
    pub confidence_db: Option<f64>,
    pub reason: Option<String>,        // stable machine token, §9
}
pub struct SourcePosition {
    pub source_id: String,
    pub position_m: Option<[f64; 3]>,  // acoustic centre, chirp band
    pub sigma_m: Option<[f64; 3]>,
    pub rms_residual_m: Option<f64>,
    pub gdop: Option<f64>,
    pub mirror_ambiguous: bool,
    pub declared_error_m: Option<f64>, // vs §8.1 surveyed position
    pub reason: Option<String>,
}
```

Conventions copied from the house style: missing stays `None` with a
reason (never 0, never omitted silently — cf. `CaptureClockProvenance`
doc comments), `#[serde(default)]` on every new field so old
`capture-clock.json` files still load, vectors in plan order, and the
report is `SupportingEvidence` in the handoff inventory (§8.4).

### 8.4 Handoff and RoomEQ (Phase 5 only)

`publish_capture_handoff` (`src/capture_session/handoff.rs:91+`) adds
`capture-clock.json` (already inventoried, and the report lives inside
it — no new file) and, in the canonical `recordings.json` projection
(`manifest.rs:48+` `recording_configuration`), each speaker gains an
optional `estimated_position` block mirroring `SourcePosition` plus the
full range row. Per §12 decision 2, AutoEQ consumes this block as an
**input to delay alignment** (a consumed value with stricter gates,
not display-only evidence): it must still never silently replace
declared geometry — out-of-tolerance estimates refuse and fall back
to declared values with a named reason. That contract needs the autoeq
owner's sign-off before Phase 5 starts; Phases 1–3 are entirely
inside this crate plus the pure `math-dsp` module.

## 9. Quality gates (fail-closed, per cell / per source)

| Gate | Condition | Effect | Reason token |
| --- | --- | --- | --- |
| Q1 clock | take basis `Uncorrected`, or pooled-skew outlier | cell refused | `clock_uncorrected` / `clock_outlier` |
| Q2 chirp confidence | arrival confidence < 6 dB (`MIN_TDOA_CONFIDENCE_DB`) | cell refused | `arrival_low_confidence` |
| Q3 estimator agreement | IR vs matched-filter arrivals differ > 1.0 sample | cell excluded from solve, kept flagged in matrix | `arrival_ambiguous` |
| Q4 plausibility | range < 0, > room diagonal + 5 m, or non-finite | cell refused | `range_implausible` |
| Q5 origin | Mode C | all absolutes `None`, TDOA only | `no_absolute_time_origin` |
| Q6 emitter anchor | measured vs surveyed emitter range off > 1 sample | session falls to Mode C | `emitter_mismatch` |
| Q7 solve residual | final RMS residual > max(1 sample in metres, 3·median σ) | position refused, ranges kept, RMS+GDOP published as evidence | `solve_residual_high` |
| Q7b solver failure | outlier depletion or divergence after the gates pass | position refused, ranges kept | `solve_failed` |
| Q8 geometry | §6.4 rank/planarity failures | position refused or mirror-flagged | `degenerate_geometry` / `mirror_ambiguous` |
| Q8b range count | fewer than 4 usable ranges (other than the 3-mic mirror case) | position refused | `insufficient_ranges` |
| Q9 SNR | take's band SNR below the coherent threshold (existing 30 dB policy findings) | cell refused | `band_snr_low` |

Thresholds are named constants beside the code that uses them (the
`COHERENT_TIMING_LIMIT_US` pattern), each with the one-line physical
justification in a comment. Changing a threshold requires updating this
table in the same commit.

## 10. CLI and user experience

- `plan-validate`: after the existing timing-emitter line
  (`timing_emitter_summary`, `src/capture_session.rs:630-648`), print a
  localization readiness block: mic count, geometry rank/coplanarity,
  mode that *would* apply (loopback declared? emitter surveyed?),
  and the §6.4 warnings ("4 microphones are coplanar: heights will be
  mirror-ambiguous — raise one microphone"). No device access.
- `plan-process`: one range line per source plus one position line after
  each row that carries a position entry, e.g.
  `localization left: 4/4 ranges 2.10–3.40 m` followed by
  `localization left position: [2.310, -1.100, 0.700] m rms 3.0 mm gdop 2.1`
  or `localization left position: refused (solve_residual_high) rms 41.2 mm`.
  A coplanar solve appends ` [mirror-ambiguous]`. Printed next to
  `clock_skew_lines` output; warnings to stderr like the existing clock
  warnings. Exit status unchanged — localization refusal is evidence,
  not a processing failure.
- No new subcommand in v1. A later `plan-localize --raw …` re-run tool
  is only justified if users need to re-solve old sessions with new
  calibrations; the report-in-manifest design already allows an
  offline re-solve without re-recording.

## 11. Implementation phases

Each phase ends with `just check`, `just clippy`
(`--all-targets --all-features --tests`), and the tests named in §13
green. No phase changes acquisition behaviour.

**Phase 0 — fixtures and bench protocol (no production code).**
Synthetic golden fixture generator (known positions, fractional delays
via the Hann-sinc FIR oracle pattern from `capture_tdoa.rs` tests,
pooled clock, known `L`, added reflections/noise at set levels);
§5.4 bench procedure written into `docs/` and executed once on the
reference rig to obtain real `L` / `b_common` magnitudes. Acceptance:
golden files committed; bench numbers recorded in the fixture header.

**Phase 1 — clock fit refactor + range matrix (Modes A/B/C, no solver).**
§5.2 split of `correct_capture_clock` (behaviour bit-identical);
`localization.rs` in `capture_session` computing §§7.2 steps 1–6;
`LocalizationReport.ranges` published; CLI lines for ranges. This
phase alone delivers Goal G1: automatic distances, the headline ask.
Acceptance: §13.1–13.3 tests; existing clock/analysis tests untouched
and green.

**Phase 2 — 3D solver.**
`math-dsp::capture_localization` (§8.2, with its own unit tests in the
math-audio ladder: `cargo check`, `clippy`, crate tests there);
session wiring for `SourcePosition`; `plan-validate` geometry
warnings. Acceptance: §13.2 solver tests including mirror/coplanar
refusals; Monte-Carlo accuracy test (§13.2) inside its stated bounds.

**Phase 3 — hardening.**
Repeat-median arrivals, sub-band (low-frequency) estimator, per-source
delay calibration input, leave-one-out diagnostics surfaced in the
report, `docs/` user section (rig setup: mic height diversity, why a
surveyed emitter or loopback matters). Acceptance: §13.4 hardware
test on the reference rig with taped ground truth.

**Phase 4 — inverse and joint modes (optional, separately approved).**
Microphone localization from surveyed sources (requires the §8.1
optional mic-position migration), then joint self-calibration
(alternating solve over sources+mics with 3 surveyed anchor points
fixing the frame; scale fixed by `c`). Each lands only with its own
identifiability analysis appended to §6.4.

**Phase 5 — RoomEQ handoff (cross-repo).**
§8.4 projection + autoeq ingestion as a delay-alignment input (§12.2),
following the frozen-contract process of `docs/plan-capture-roomeq-handoff.md`:
agree the schema with the roomeq owner first, pin with a cross-repo
handshake test like `emitted_coherence_sidecar_loads_into_autoeq_curve`.

## 12. Decisions (resolved 2026-10-14)

1. Reference rig truth: **Mode A** — one interface in+out with the plan
   loopback wired. Phase 0 bench work and the §13.4 hardware test run
   on this rig; the surveyed emitter is off the critical path.
   (Mode B/C support stays in the plan as designed.)
2. `autoeq` will consume estimated positions **as an input to delay
   alignment**, not display-only evidence. §8.4 is therefore a consumed
   value with stricter gates (to be specified with the autoeq owner in
   Phase 5: tighter residual/σ thresholds than the display-only case).
3. Mic-count ceiling: **up to 32 microphones** (owner has 8). The plan
   schema cap of 4 (`src/capture_session.rs:352-354`) and the take-count
   limit (1024) must be raised accordingly: 32 mics × sources × repeats
   must fit, or per-source/per-repeat limits must be added. This touches
   the RoomEQ matrix contract — coordinate with the autoeq owner; the
   solver itself is mic-count agnostic. Minimum for 3D positioning
   stays at 4 non-coplanar mics (§6.4).

## 13. Test plan

### 13.1 Unit — DSP layer (math-dsp)

- Solver recovers known positions from exact ranges (residual < 1e-9).
- Solver with Gaussian range noise (σ = 5 mm, fixed seed): position
  error within 3× the predicted covariance σ over 200 trials.
- Coplanar mics ⇒ `mirror_ambiguous`; collinear ⇒ `None`; 2 mics ⇒ `None`.
- Huber/IRLS: one range corrupted by +15 cm still yields position
  within 2 cm and flags the cell as excluded.
- Conversion uncertainty (`propagation_to_meters`, `range_sigma_m`,
  session layer): unknown inputs ⇒ `None` (unknown sigma), never zero.

### 13.2 Unit — session layer (this crate)

Golden-fixture tests modelled on `clock/tests.rs`: synthesize takes
with the FIR/`sample_clock` oracle at known offsets, skews, and
propagation delays for known geometry, then assert recovered ranges
within 0.2 sample in metres and positions within the §13.1 bounds.
Cases: Modes A, B, C (C asserts all absolutes `None`); one mic's
chirp buried in noise → only its cells refused; a +8-sample reflection
stronger than direct → Q3 flags the cell, position still solved from
the rest; sub-band source; 3 repeats with one wild arrival → median
wins. Serde: old `capture-clock.json` without the new field loads;
old plans without the new optional keys validate.

### 13.3 Regression

Existing clock golden tests, `capture_session` tests, signal-recorder
tests, and `just check/clippy/test` all green with no expectation
changes. The §5.2 refactor is guarded by asserting byte-identical
`capture-clock.json` on the fixture session before/after.

### 13.4 Hardware (reference rig, once per phase ≥ 1)

Four matched mics on one interface, positions taped/lasered to ±2 mm
with deliberate height diversity (e.g. 0.9–1.6 m); two speakers at
taped positions; loopback wired. Assert: ranges within 2 cm of tape,
positions within 4 cm per axis, and the emitter-anchor check (Q6)
passes. Record the achieved numbers back into §6.3's table — the
budget is updated from measurement, not aspiration. A second run with
all mics at one height must print the mirror warning and set the flag.

## 14. Future extension: leaving the single-interface assumption

If multi-device microphone rigs are ever required, the changes are
localized by design: each device regains an offset (and skew) unknown,
estimated per device from that device's view of the fixed emitter —
the emitter stops being optional and becomes the per-device time
origin (Mode B generalized per device); `b_mic` becomes per-unit,
requiring per-unit bench calibration instead of one `b_common`; and
takes are no longer simultaneous across devices, so source position
solves must tolerate sequential, drift-corrected evidence with the
per-take uncertainty the clock stage already models. The range/position
report format does not change — only the mode enum and the gates gain
members. None of this is built, stubbed, or pre-abstracted in v1.

## 15. Risks

| Risk | Likelihood | Mitigation in this plan |
| --- | --- | --- |
| Reflection stronger than direct sound (speaker near wall, mic off-axis) | High in real rooms | Dual-estimator gate Q3, Huber down-weighting, per-cell exclusion, repeats median |
| Speaker DSP/AVR latency mistaken for distance | Medium | §5.5 explicit calibration field; output labelled acoustic-centre; no assumed defaults |
| Subwoofer silent in chirp band | Certain for subs | §4.3 low-band estimator + widened σ, band recorded per cell |
| User surveys mics coplanar (seat row) and expects heights | High | `plan-validate` warning before recording; mirror flag on output |
| Sound speed taken as exact 343 | High if unmeasured | §6.2 uncertainty propagates into σ; temperature input supported |
| Clock refactor regresses coherent capture | Low, guarded | §13.3 byte-identical fixture assertion; existing golden tests unmodified |
| Report schema breaks old bundles | Low | serde defaults everywhere; old-file load tests (§13.2) |

## 16. References (existing code this plan builds on)

- Stimulus/routing: `src/capture_session/protocol.rs`,
  `src/capture_session/record.rs:598-620` (chirp rerouting to emitter).
- Clock fit and bases: `src/capture_session/clock.rs` (esp. 205-248),
  uncertainty model `math-dsp/src/capture_tdoa.rs:283-354`.
- TDOA primitive: `math-dsp/src/capture_tdoa.rs:91-229`
  (`estimate_chirp_tdoa`, confidence semantics).
- Array geometry/DOA precedent: `math-dsp/src/capture_array.rs`
  (`MicArray`, `estimate_doa_ls`, `pairwise_tdoas_vs_first`),
  `src/capture_session/reflections.rs` (mirror-ambiguity honesty).
- IR/delay preservation: `src/capture_session/phase.rs` and its
  `exported_ir_retains_delay…` test.
- Processed-manifest and CLI plumbing: `src/capture_session/clock/io.rs`,
  `src/bin/sotf_capture.rs:871-934`.
- Handoff contract process: `docs/plan-capture-roomeq-handoff.md`.
