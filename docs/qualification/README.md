# Capture callback and storage qualification

Run the hardware-free callback matrix in release mode:

```sh
cargo bench --locked --offline --features test-util --bench capture_callbacks -- --iterations 4096 --rate 48000 > callbacks.jsonl
```

The benchmark exercises the production sweep input core with its level meter,
auxiliary input with electrical loopback, multi-device input/output, and scheduled
auxiliary output. It uses all twelve supported PCM representations: f32/f64 and
signed/unsigned 8-, 16-, 24-, 32-, and 64-bit integers. Hardware widths are 2, 8,
33, and 94, with nominal callback blocks of 32, 128, 512, and 1024 frames. Input
probes include healthy delivery, full-ring overflow, and malformed blocks for
every PCM type, plus late-block NaNs for both float types: 2,208 cases total.
Malformed fixtures omit one scalar from the nominal block; their deadline still
uses its configured nominal frame count. NaNs occupy the final selected/hardware
channels of the final frame, exercising a valid scan prefix before rejection.
The multi-device rejection control drains its retained valid prefix so repeated
measurements cannot silently become overflow measurements. Input ring draining and output
cursor resets run outside the timed callback. Warmup, buffer allocation, result
storage allocation, sorting, and reporting are excluded from timing. The sampled
allocation region includes callback and control work; any allocation,
reallocation, or deallocation fails the run. Instrumentation uses `stats_alloc`
in the benchmark executable; production applications retain their own allocator.
The measured control work is ring draining and cursor reset, excluding native
health-error formatting and disk/report publication. Input buffers and playback
objects pass through optimizer barriers inside the callback closures so constant
fixture data cannot be processed ahead of the measured interval.

Each JSON line identifies the case and reports median, p99, and maximum callback
time, its nominal frame deadline, and heap-operation count. Use
`--max-deadline-fraction 0.5` to make a maximum latency above half the deadline
fail the command. This threshold is optional because scheduler preemption and
machine load affect maximum wall time. Choose and record a threshold for the
qualification machine; an unconfigured run enforces heap behavior only.

On Apple/Linux/Android hosts, add `--cpu-clock thread` for paired thread CPU
diagnostics. CPU time excludes periods when the thread is descheduled. Its
measurement brackets the wall interval and includes timer-call overhead, so
`wall_minus_cpu_lower_bound_ns_at_max_wall` is a conservative lower bound on
wall time without CPU execution, not an exact scheduler trace. CPU samples do
not replace wall time in the deadline gate. Unsupported platforms reject this
option instead of substituting another clock. Reports without this option emit
null CPU fields. For example:

```sh
cargo bench --locked --offline --features test-util --bench capture_callbacks -- --iterations 4096 --rate 48000 --max-deadline-fraction 1 --cpu-clock thread
```

For memory measurement, first build the executable:

```sh
cargo bench --locked --offline --features test-util --bench capture_callbacks --no-run
```

Cargo prints the executable path. Invoke that executable directly with
`--memory-only`. On macOS, wrap it in `/usr/bin/time -l`; on GNU/Linux, use
`/usr/bin/time -v`. Measuring `cargo` would also measure compiler processes and
would not isolate the probe. For example, substitute the printed executable:

```sh
/usr/bin/time -l /path/to/capture_callbacks-HASH --memory-only > storage.json 2> storage-rss.txt
```

This probe holds 33 selected inputs at 48 kHz, 3,480,000 retained frames per
input, selected-channel rings, and two 3,000,000-frame stimulus buffers. It uses
the production recording-buffer allocator and ring constructor and writes every
payload page before sampling. It reports the source budget and measured live
allocator bytes. The representative allocation must remain below 512 MiB; this
does not prove that every accepted plan has the same RSS. Native driver buffers,
device construction, and later DSP/repeat analysis are excluded. RSS also includes
the executable, allocator overhead, and runtime. Keep allocator bytes and RSS
separate in reports.

Native session recording now uses the file-backed path automatically. Direct
`capture_multidevice` callers retain the bounded in-memory API;
`capture_multidevice_to_files` returns finalized private WAVs for explicit
publication. Disk writes run on a dedicated worker. The control thread sends
1,024-frame mono batches without waiting for I/O, and the bounded queue holds
one nominal second across the selected inputs (at least 64 batches). A full
queue or a writer error fails acquisition. Streams stop before finalization;
finalization/cleanup may still wait for operating-system filesystem operations.
Failed/cancelled captures do not publish their private files. Already completed
sources remain available in a failed/cancelled session's nonterminal journal.

Measure the file-backed implementation with the printed release executable:

```sh
/usr/bin/time -l /path/to/capture_callbacks-HASH --memory-only --storage files > file-storage.json 2> file-storage-rss.txt
```

This probe uses the maximum software configuration: 33 inputs, 384 kHz, and
16,000,000 frames per input. It writes 2,112,000,000 bytes of temporary WAV sample
payload, finalizes every header, and reads every sample back before deleting the
files. The source budget includes selected-channel rings, two stimulus buffers,
pending batches, the rate-scaled writer queue, buffered writers, and bookkeeping.
The producer waits between batches to accelerate a hardware-free memory probe;
this does not qualify physical input timing or sustained native disk throughput.
The files require approximately 2 GiB of temporary disk space during the run.
File-writer/control allocations are permitted and bounded; the zero-heap gate
applies to the callback-core matrix, not the disk worker.

Native file-backed acquisition failures also write `capture-loss-*.json` after
streams stop. Each input clock retains at most 128 missing intervals, an exact
total missing-frame count per selected channel, and an explicit omitted-interval
count. Reports preserve actual device IDs and logical/hardware channel routes.
Positions start at each input's first callback frame; they are not stimulus-clock
offsets. Malformed blocks or asynchronous input errors carry an uncertainty flag.
The error and failed/cancelled session journal name the report, or record why it
could not be saved. Standalone sweep and auxiliary recording APIs now save
equivalent reports beside their requested WAVs. `SweepRecorder::transport_evidence`
returns typed snapshots after stop; native snapshots preserve the actual input
device ID, channel, and negotiated rate until the next start attempt. APIs
without a recording destination append bounded JSON diagnostics to ordinary
failure errors. Standalone cancellation retains the exact `cancelled` marker;
its saved report remains discoverable by filename, and report-write failures
are logged on the control thread. Invalid blocks advance known whole-frame
positions while retaining an explicit uncertainty flag. Sweep output identity
is left null when the backend cannot attest it; the requested selector is saved
separately. Input-clock positions are never substituted for stimulus offsets.
Raw journals now declare these reports in `transport_evidence`, using exact
filenames and SHA-256 hashes. Recording registers reports only in its newly
created capture directory. Processing existing journals reads only declared
reports, verifies and freezes their bytes before creating output, and retains
them in the moved handoff inventory. Missing, changed, unsafe, malformed, or
oversized reports prevent publication. Limits are 512 reports, 1 MiB per report,
and 8 MiB in aggregate. Existing journals default to an empty list; unrelated
files are never discovered during import, even if their names match the prefix.
New journals also save `checkpoint` phase/completion maps before each acquisition
pass. Interrupted passes retain their source/repeat or auxiliary phase; committed
take IDs, loopback passes, and probe/anchor flags must agree with saved results.
Processing rejects contradictory checkpoints and incomplete declared passes.
Legacy journals default to no checkpoint and retain the ordinary completeness
checks. A phase identifies attempted work and does not certify hardware delivery.
The file-storage probe also holds 33 independent clock diagnostics to include
their bounded allocation in the measurement.

Record the OS, architecture, CPU, Rust version, exact command, source revisions
and local changes, and other machine load beside results. These probes do not
establish physical device latency, timestamp accuracy, worst-case scheduler
behavior, or hard realtime safety. Format/channel combinations exercise typed
cores and do not assert that a particular physical device advertises every
combination. Native stream-error delivery, including CPAL error ownership, and
native callback entry/exit remain outside this matrix.

`cargo test --features test-util --bench capture_callbacks` runs one callback and
control step per case as a smoke check. Debug runs do not emit performance data.

The separate blocking-logger watchdog runs with:

```sh
cargo test --locked --offline --features test-util --test callback_no_log
```

It enables Trace logging in an isolated child, verifies a deliberate log call
really blocks, and kills that positive-control child. A fresh child then invokes
every healthy, overflow, malformed, and non-finite callback case with the blocking logger installed. The
parent requires successful completion without logging and terminates any hung
child. Its ten-second process bound accommodates startup on loaded hosts; it is
not an audio deadline assertion. These tests exercise callback cores and control
steps, excluding native device construction and callback timestamp delivery.
