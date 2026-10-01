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
