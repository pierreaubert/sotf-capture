//! Opt-in allocation and latency qualification of production capture callback cores.

// Rust guideline compliant 2026-02-21

use clap::Parser;
use sotf_capture::signal_recorder::qualification::{
    visit_callbacks, visit_file_storage, visit_storage,
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
use std::time::Instant;

#[global_allocator]
static ALLOCATOR: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum CpuClock {
    Thread,
}

#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
enum StorageKind {
    #[default]
    Memory,
    Files,
}

#[derive(Debug, Parser)]
struct Args {
    /// Cargo supplies this flag to custom benchmark harnesses.
    #[arg(long, hide = true)]
    bench: bool,
    /// Cargo test asks custom benchmark harnesses for a smoke run.
    #[arg(long, hide = true)]
    test: bool,
    /// Callback samples per case; each case also receives 32 warmup iterations.
    #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..=1_000_000))]
    iterations: u32,
    /// Rate used to calculate the callback's scheduling deadline.
    #[arg(long, default_value_t = 48_000, value_parser = clap::value_parser!(u32).range(8000..=384_000))]
    rate: u32,
    /// Fail if the maximum sampled callback exceeds this fraction of its deadline.
    #[arg(long)]
    max_deadline_fraction: Option<f64>,
    /// Add paired thread-CPU diagnostics; the deadline gate still uses wall time.
    #[arg(long, value_enum)]
    cpu_clock: Option<CpuClock>,
    /// Measure touched acquisition storage without running the callback matrix.
    #[arg(long)]
    memory_only: bool,
    /// Storage implementation measured with --memory-only.
    #[arg(long, value_enum, default_value_t)]
    storage: StorageKind,
}

fn main() {
    let args = Args::parse();
    // Cargo test runs a harness=false benchmark without arguments. Its debug
    // invocation is a correctness smoke run, never a performance measurement.
    if args.test || (cfg!(debug_assertions) && std::env::args_os().len() == 1) {
        visit_callbacks(|_, callback, control| {
            callback();
            control();
        });
        return;
    }
    if cfg!(debug_assertions) {
        eprintln!("run with cargo bench (release profile)");
        std::process::exit(2);
    }
    if let Some(limit) = args.max_deadline_fraction {
        assert!(
            limit.is_finite() && limit > 0.0,
            "deadline fraction must be positive and finite"
        );
    }
    if args.memory_only {
        let region = Region::new(ALLOCATOR);
        if matches!(args.storage, StorageKind::Files) {
            visit_file_storage(|case| {
                let change = region.change();
                let live_bytes = change.bytes_allocated - change.bytes_deallocated;
                assert!(live_bytes <= case.budget_bytes);
                println!("{}", serde_json::json!({
                    "storage": case, "live_allocated_bytes": live_bytes,
                    "allocations": change.allocations, "reallocations": change.reallocations,
                    "scope": "file-backed acquisition, including rings, writer, and stimuli; excludes native devices and DSP analysis"
                }));
            }).unwrap();
            return;
        }
        visit_storage(|case| {
            let change = region.change();
            let live_bytes = change.bytes_allocated - change.bytes_deallocated;
            // This representative valid case leaves room for ring/Vec metadata
            // within the acquisition cap. RSS is measured externally, separately.
            assert!(live_bytes <= 512 * 1024 * 1024);
            println!("{}", serde_json::json!({
                "storage": case,
                "live_allocated_bytes": live_bytes,
                "allocations": change.allocations,
                "reallocations": change.reallocations,
                "scope": "selected acquisition storage; excludes native devices and DSP analysis"
            }));
        }).unwrap();
        return;
    }
    if args.cpu_clock.is_some() && thread_cpu_ns().is_none() {
        eprintln!("thread CPU clock is unavailable on this platform");
        std::process::exit(2);
    }
    run_callbacks(&args);
}

fn run_callbacks(args: &Args) {
    let mut exceeded = false;
    visit_callbacks(|case, callback, control| {
        let mut times = Vec::with_capacity(args.iterations as usize);
        let mut cpu_times = args
            .cpu_clock
            .map(|_| Vec::with_capacity(args.iterations as usize));
        let mut cpu_at_max_wall = None;
        let mut max_wall = 0;
        for _ in 0..32 {
            callback();
            control();
        }
        let region = Region::new(ALLOCATOR);
        for _ in 0..args.iterations {
            let cpu_start = args.cpu_clock.and_then(|_| thread_cpu_ns());
            let start = Instant::now();
            callback();
            let wall_ns = start.elapsed().as_nanos();
            let cpu_ns = cpu_start.and_then(|start| thread_cpu_ns()?.checked_sub(start));
            if wall_ns >= max_wall {
                max_wall = wall_ns;
                cpu_at_max_wall = cpu_ns;
            }
            times.push(wall_ns);
            if let Some(samples) = &mut cpu_times {
                samples.push(cpu_ns.expect("validated thread clock must remain monotonic"));
            }
            control();
        }
        let change = region.change();
        assert_eq!(
            change.allocations, 0,
            "callback/control allocated: {case:?}"
        );
        assert_eq!(
            change.reallocations, 0,
            "callback/control reallocated: {case:?}"
        );
        assert_eq!(
            change.deallocations, 0,
            "callback/control deallocated: {case:?}"
        );
        times.sort_unstable();
        if let Some(samples) = &mut cpu_times {
            samples.sort_unstable();
        }
        let deadline_ns =
            u128::try_from(case.frames).unwrap() * 1_000_000_000 / u128::from(args.rate);
        let max_ns = times[times.len() - 1];
        #[expect(
            clippy::cast_precision_loss,
            reason = "Ratios summarize bounded nanosecond samples for an informational benchmark report."
        )]
        let fraction = max_ns as f64 / deadline_ns as f64;
        exceeded |= args
            .max_deadline_fraction
            .is_some_and(|limit| fraction > limit);
        println!(
            "{}",
            serde_json::json!({
                "case": case,
                "iterations": args.iterations,
                "rate": args.rate,
                "p50_ns": times[(times.len() - 1) / 2],
                "p99_ns": times[(times.len() - 1) * 99 / 100],
                "max_ns": max_ns,
                "deadline_ns": deadline_ns,
                "max_deadline_fraction": fraction,
                "thread_cpu_p50_ns": cpu_times.as_ref().map(|samples| samples[(samples.len() - 1) / 2]),
                "thread_cpu_max_ns": cpu_times.as_ref().map(|samples| samples[samples.len() - 1]),
                "thread_cpu_ns_at_max_wall": cpu_at_max_wall,
                "wall_minus_cpu_lower_bound_ns_at_max_wall": cpu_at_max_wall.map(|cpu| max_ns.saturating_sub(cpu)),
                "heap_operations": change.allocations + change.reallocations + change.deallocations,
            })
        );
    });
    assert!(
        !exceeded,
        "callback deadline threshold exceeded; inspect JSON results"
    );
}

// Read the current thread's CPU usage, which excludes time descheduled. Its
// sample brackets the wall interval and therefore includes timer-call overhead.
#[cfg(any(target_vendor = "apple", target_os = "linux", target_os = "android"))]
fn thread_cpu_ns() -> Option<u128> {
    let time = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
    u128::try_from(time.tv_sec)
        .ok()?
        .checked_mul(1_000_000_000)?
        .checked_add(u128::try_from(time.tv_nsec).ok()?)
}

#[cfg(not(any(target_vendor = "apple", target_os = "linux", target_os = "android")))]
fn thread_cpu_ns() -> Option<u128> {
    None
}
