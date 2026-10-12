# sotf-capture

Acoustic measurement capture for the SOTF ecosystem: swept-sine and probe
recording, clock-drift correction, take quality gating, capture sessions,
and the recording-wizard domain model. RoomEQ optimization stays in
`autoeq`; the DAW engine stays in `sotf-daw`; both depend on this crate.

Part of [SotF](https://github.com/pierreaubert/sotf): extracted from `sotf-player` 
(`recording_types`, `recording_helpers`, `capture_session`, `RecordingScreenModel`)
and `sotf-engine` (`signal_recorder`, `devices`, `rate_limit`) so capture 
has one owner.

## Layout

- `recording_types` / `recording_helpers`: capture types and session JSON.
- `capture_session`: multi-capture pipeline (protocol/record/clock/phase).
- `wizard`: UI-agnostic `RecordingScreenModel` shared by GPUI/TUI/CLI.
- `signal_recorder`: stimulus generation, play/record takes, analysis.
- `signal_recorder::playback`: output backends behind `SweepPlayback`.
- `devices`: audio device lookup (stubbed on iOS).

## Playback backends

`record_and_analyze_with` / `record_and_analyze_with_multi` take a
`SweepPlayback` backend. `CpalPlayback` (here) plays through cpal
directly; the engine backend lives in `sotf-engine` so existing
frontends keep identical playback behavior. New callers use cpal.

## Build

```bash
just check
just clippy
just test
just dev
just prod
```

## Saved acquisition handoff

Clock processing retains original device-clock WAV snapshots with processed
audio, responses and calibration snapshots. When a canonical `recordings.json`
projection is available, the final `capture-handoff.json` inventory binds all
files by SHA-256 and records source/microphone/take identity, rate, acquisition
status, device, gain, position and clock declarations.

New configurations require this inventory when loaded by RoomEQ. Move the whole
processed directory together. Changed or missing files are refused, and parsed
responses are frozen before optimization. Calibration is applied during capture
analysis and is not reapplied by the handoff loader. Hashes do not authenticate
hardware calibration or establish listening benefit. Repeated/partial capture
selection remains an explicit review step; missing clocks remain unknown.

## RoomEQ handoff rules

One plan per optimization. Sources that RoomEQ combines coherently — subs and
mains for bass alignment, channels for Group-Delay optimization — must come
from a single `plan-record`/`plan-process` run: one sample rate, one sweep
layout, one shared timing reference, one response grid. Separate process runs
are never merged for coherent use; their grids and clocks are unrelated even
when the plans look identical, and GD-Opt skips mismatched grids silently by
design. The processed directory carries everything RoomEQ needs: room
dimensions when the plan declares them, per-source room IRs from the selected
IR microphone, per-bin coherence beside each phase response, the resolved
timing identity, probe/anchor passes when recorded, and per-take IR-only STI.

## Session plans

A session plan is one JSON file describing everything RoomEQ needs later:
microphone routes and positions, sources, the sweep, the timing reference,
room dimensions, and optional probe/bass-anchor/CTC passes. Record with a
plan whenever the takes will feed `autoeq`/`roomeq`; the one-shot `capture`
command is for quick checks only. Runnable examples live in `examples/`:

| Plan | Shape |
| --- | --- |
| `stereo-two-seats.json` | Minimal stereo room: 2 sources, 2 seat mics, room dims |
| `home-theater-5.1-sub.json` | Full rig: 6 sources, dedicated timing emitter, probe + bass-anchor + loopback passes, bass-weighted sweep |
| `ctc-head-position.json` | Crosstalk-cancellation ear pair: compact geometry, loopback, one head position |
| `full-roomeq-session.json` | Everything in one run: seats + ears, probe + bass-anchor + CTC, dedicated timing emitter |

Workflow:

```bash
sotf-capture plan-validate --plan examples/stereo-two-seats.json
sotf-capture plan-record --plan my-room.json --output raw-2026-10-11
sotf-capture plan-process --raw raw-2026-10-11/capture-raw.json --output processed-2026-10-11
```

A minimal plan looks like this (see `examples/stereo-two-seats.json` for the
full file):

```json
{
  "version": 1,
  "geometry": "spread",
  "sample_rate_hz": 48000,
  "output_device": "ADAM Audio USB",
  "microphones": [
    {
      "id": "main-seat",
      "device": "UMIK-1 A",
      "input_channel": 0,
      "calibration_file": "calibrations/umik-a-90deg.txt",
      "calibration_orientation": "ninety_degrees",
      "gain_db": 0.0,
      "position_m": [0.0, 0.0, 1.2],
      "position_uncertainty_mm": 5.0
    },
    {
      "id": "second-seat",
      "device": "UMIK-1 B",
      "input_channel": 0,
      "calibration_file": "calibrations/umik-b-90deg.txt",
      "calibration_orientation": "ninety_degrees",
      "gain_db": 0.0,
      "position_m": [0.6, 0.0, 1.2],
      "position_uncertainty_mm": 5.0
    }
  ],
  "sources": [
    {"id": "left", "output_channel": 0},
    {"id": "right", "output_channel": 1}
  ],
  "sweep": {"duration_secs": 5.0, "start_hz": 20.0, "end_hz": 20000.0, "amplitude": 0.1},
  "room_dimensions": {"length": 5.2, "width": 3.8, "height": 2.6}
}
```

Rules the validator enforces:

- Two to four microphones, each with a unique id, a unique
  device/channel route, and its own calibration file resolved relative to
  the plan. Device names must match `sotf-capture devices` output.
- At least one source; every source gets its own output channel and is
  excited separately while all microphones record.
- Without `timing_reference`, timing chirps ride on the first source
  channel and takes stay arrival-aligned; declare a fixed emitter (with
  surveyed position and sound speed) for coherent use such as sub/mains
  alignment or group-delay optimization.
- `bass_anchor` requires a `loopback` route on the first microphone's
  device; `ctc` requires a `loopback` route plus two distinct ear
  microphones. `room_dimensions` (meters) and `setup_description` project
  straight into the RoomEQ input format.
- `compact` geometry (ear pairs, arrays) requires positions surveyed to
  1 mm and distinct per microphone.

Before the full session, run `live` to check levels and one planned take
at a conservative level: confirm routing, peaks, reference lock, and a
plausible response, then commit to the whole run.

## Per-machine device and calibration parameters

Choose input and output independently with `sotf-capture capture --input-device
<MIC_DEVICE_ID> --output-device <INTERFACE_DEVICE_ID>`, using IDs or unique names
from `sotf-capture devices --json`. Set `--hwaudio-record-from` and
`--hwaudio-send-to` to the actual zero-based hardware channels. The legacy
`--device` selector assigns one device to both directions and conflicts with
the independent selectors. Omitted selectors retain the existing system-default
behavior; JSON summaries record requested selectors separately from device
negotiation evidence.

Assign each input's calibration explicitly with repeatable
`--mic-calibration CH=PATH`; duplicate, empty or unused assignments are refused.
Reusable calibration files are available under `data_tests/microphones`
(`../sotf-capture/data_tests/microphones` from the AutoEQ checkout). Select the
correct microphone serial and orientation; directory membership alone does not
identify the connected device or supply absolute SPL calibration. Session-plan
fields preserve per-microphone orientation, gain and geometry. RME, UMIK and
playback-system names are runtime choices and are never fixed fixture identities.

## Raw live level and RTA

`sotf-capture live --input-device <EXACT_ID> --input-channel 0 --sample-rate
48000 --fft-size 4096 --duration 30` emits versioned JSON `live_frame` events
and a final `live_stopped` summary. Ctrl-C cancels and releases the input stream.
The monitor plays no stimulus. Input rate and channel must be supported exactly;
ambiguous names and fuzzy selectors are refused.

The callback keeps at most four analysis blocks in a lock-free ring. Slow
consumers cause counted drops; sample sequence identities ensure a later frame
containing the gap is invalid even after older buffered samples have drained.
Analysis runs outside the callback. The summary records observed update rate,
received/dropped samples and excluded partial samples.

RMS and sample peak are raw dBFS. RTA bins are symmetric-Hann, one-sided peak
amplitude dBFS with no overlap or padding, using the existing math-dsp analyzer.
These are amplitude bins rather than PSD or band-integrated noise levels.
Absolute SPL is `null`; microphone response compensation and absolute calibration
remain separate from this raw monitoring path. Digital silence has no finite
RMS/peak dB value. Nonfinite input and gaps have empty spectra and explicit flags.

## SPL calibration anchors

The SPL calibration backend accepts an unclipped, nonzero finite RMS reference
capture plus a finite contemporaneous external-meter reading. Silence, invalid
levels, invalid rates and reference frequencies at or above Nyquist are refused.
Starting a new calibration invalidates the previous capture and meter reading;
only completed captures can supply an anchor in the recording configuration.

The offset is `meter dBSPL - 20 log10(captured RMS)`. Reuse requires the same
microphone, input channel and gain. It does not supply frequency-response
compensation or calibration for a different machine, and the raw live monitor
continues to report unknown absolute SPL.
