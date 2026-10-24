//! Verify callback cores cannot block on a logger, with a bounded child watchdog.

// Rust guideline compliant 2026-02-21

use sotf_capture::signal_recorder::qualification::visit_callbacks;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

const CHILD_ROOT: &str = "SOTF_CALLBACK_LOG_TEST_ROOT";
const CHILD_MODE: &str = "SOTF_CALLBACK_LOG_TEST_MODE";
// Give heavily loaded hosts time to launch a test process. This is a watchdog
// bound, not an audio callback deadline or a performance assertion.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(10);

struct BlockingLogger(PathBuf);

impl log::Log for BlockingLogger {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, _: &log::Record<'_>) {
        std::fs::write(&self.0, b"logger entered").unwrap();
        loop {
            std::thread::park();
        }
    }

    fn flush(&self) {}
}

struct WatchedChild(Child);

impl WatchedChild {
    fn start(root: &Path, mode: &str) -> Self {
        Self(
            Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg("callbacks_cannot_block_on_a_logger")
                .env(CHILD_ROOT, root)
                .env(CHILD_MODE, mode)
                .stdout(Stdio::null())
                .spawn()
                .unwrap(),
        )
    }

    fn wait(&mut self, timeout: Duration) -> Option<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                return Some(status);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for WatchedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn callbacks_cannot_block_on_a_logger() {
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        let marker = PathBuf::from(root).join("logger-entered");
        log::set_logger(Box::leak(Box::new(BlockingLogger(marker)))).unwrap();
        log::set_max_level(log::LevelFilter::Trace);
        assert!(
            log::log_enabled!(log::Level::Trace),
            "qualification requires Trace logging"
        );
        if std::env::var(CHILD_MODE).unwrap() == "canary" {
            log::error!("intentional watchdog canary");
            panic!("blocking logger unexpectedly returned");
        }
        visit_callbacks(|_, callback, control| {
            callback();
            control();
        });
        return;
    }

    let root = tempfile::tempdir().unwrap();
    let marker = root.path().join("logger-entered");
    // Positive control: prove logging really blocks and the parent can detect
    // it and release the process without leaving a hung test thread behind.
    let mut canary = WatchedChild::start(root.path(), "canary");
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    while !marker.exists() {
        assert!(
            canary.0.try_wait().unwrap().is_none(),
            "canary exited before logging"
        );
        assert!(Instant::now() < deadline, "canary never entered logger");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(canary.wait(Duration::from_millis(100)).is_none());
    drop(canary);
    std::fs::remove_file(&marker).unwrap();

    let mut callbacks = WatchedChild::start(root.path(), "callbacks");
    let status = callbacks
        .wait(STARTUP_TIMEOUT)
        .expect("callback/control path blocked with the slow logger installed");
    assert!(status.success(), "callback child failed: {status}");
    assert!(!marker.exists(), "callback/control path invoked the logger");
}
