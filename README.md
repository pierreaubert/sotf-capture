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

Optional calibrated live output requires both `--calibration-profile profile.json` and
`--machine-settings machine.json`. The profile stores the binding recorded at calibration
time. The machine file stores a separate `runtime_binding`; it must match every saved field
before the monitor opens the device. A changed device, gain, channel, rate, format, or
orientation requires a new calibration. Run `--validate-calibration` with both files to
check their bounded JSON and response-curve inputs without opening hardware.

Schema 1 profile files may include a response curve and an RMS SPL anchor. Curve paths are
relative to the profile file. The optional curve `sha256` checks the curve itself. An anchor
must also record `response_sha256`: use the digest of the exact curve used for that anchor,
or explicit `null` only when the anchor was made without a response curve. The loader checks
the saved anchor digest against the profile curve, so replacing curve bytes cannot silently
rebind an old absolute SPL reading. Machine `runtime_binding` repeats the saved profile
binding and adds the exact `input_selector` and optional `spl_band_hz`.

For example, start from a profile with a saved per-machine binding and fill in the selected
curve format, sign convention, orientation, and anchor values explicitly:

```json
{
  "schema_version": 1,
  "binding": {
    "microphone_id": "SELECTED_MICROPHONE_ID",
    "host_api": "SELECTED_HOST_API",
    "input_device_id": "EXACT_DEVICE_ID",
    "input_channel": 0,
    "sample_rate_hz": 48000,
    "input_sample_format": "F32",
    "declared_input_gain_db": 0.0,
    "gain_attested": true,
    "orientation": "on_axis"
  },
  "response_curve": {
    "path": "response.csv",
    "format": "csv",
    "convention": "positive_db_means_microphone_too_loud",
    "reference_frequency_hz": 1000.0,
    "sha256": "OPTIONAL_64_HEX_DIGEST"
  },
  "rms_anchor": {
    "response_sha256": "64_HEX_DIGEST_OF_THE_CURVE_USED_FOR_THIS_ANCHOR",
    "reference_rms_full_scale": 0.1,
    "reference_level_db_spl": 94.0,
    "reference_frequency_hz": 1000.0,
    "band_hz": [990.0, 1010.0],
    "weighting": "z"
  }
}
```

Machine settings use the same binding object under `runtime_binding`, plus
`{"schema_version":1,"input_selector":"EXACT_DEVICE_ID","spl_band_hz":[100.0,20000.0]}`.
The microphone calibration files in `../sotf-capture/data_tests/microphones` are inputs to
select explicitly; their sensitivity text does not create an RMS SPL anchor or determine the
response sign, serial, or orientation automatically.

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
