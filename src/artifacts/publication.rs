//! Publish complete artifact directories without replacing another writer's output.

// Rust guideline compliant 2026-02-21

use std::path::{Path, PathBuf};

/// Private sibling storage, removed on ordinary failure and atomically published.
///
/// An abrupt process exit can leave a private staging directory. It cannot make
/// that directory visible at the destination before publication. A retry uses
/// fresh staging; it never imports or deletes an abandoned directory. This is
/// process-crash isolation, not a power-loss durability guarantee.
#[derive(Debug)]
pub(crate) struct StagedDirectory {
    staging: tempfile::TempDir,
    output: PathBuf,
}

impl StagedDirectory {
    /// Create private storage beside a previously absent destination.
    ///
    /// # Errors
    /// Returns an error if the destination exists or staging cannot be created.
    pub(crate) fn new(output: &Path) -> Result<Self, String> {
        match std::fs::symlink_metadata(output) {
            Ok(_) => {
                return Err(format!(
                    "refusing to overwrite existing {}",
                    output.display()
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("cannot inspect bundle destination: {error}")),
        }
        let parent = output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        let staging = tempfile::Builder::new()
            .prefix(".capture-staging-")
            .tempdir_in(parent)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            staging,
            output: output.to_path_buf(),
        })
    }

    /// Return the private directory in which to build the complete bundle.
    pub(crate) fn path(&self) -> &Path {
        self.staging.path()
    }

    /// Atomically expose the finished directory at its destination.
    ///
    /// # Errors
    /// Returns an error if another writer created the destination or rename fails.
    pub(crate) fn publish(self) -> Result<(), String> {
        rename_directory(self.staging.path(), &self.output)
            .map_err(|error| format!("cannot publish bundle: {error}"))
    }
}

#[cfg(any(
    target_vendor = "apple",
    target_os = "linux",
    target_os = "android",
    target_os = "redox"
))]
fn rename_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        source,
        rustix::fs::CWD,
        destination,
        rustix::fs::RenameFlags::NOREPLACE,
    )
    .map_err(Into::into)
}

#[cfg(windows)]
fn rename_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
    // Windows directory rename rejects an existing destination, including an
    // empty directory. File replacement semantics do not apply to directories.
    std::fs::rename(source, destination)
}

#[cfg(not(any(
    target_vendor = "apple",
    target_os = "linux",
    target_os = "android",
    target_os = "redox",
    windows
)))]
fn rename_directory(_: &Path, _: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "atomic no-replace directory publication is unavailable on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destination_created_after_staging_is_preserved() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("bundle");
        let staging = StagedDirectory::new(&output).unwrap();
        let staging_path = staging.path().to_path_buf();
        std::fs::write(staging.path().join("member"), b"new audio").unwrap();
        std::fs::create_dir(&output).unwrap();
        assert!(staging.publish().is_err());
        assert_eq!(std::fs::read_dir(&output).unwrap().count(), 0);
        assert!(!staging_path.exists());
    }

    #[test]
    fn concurrent_publishers_have_one_winner_and_no_mixed_members() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("bundle");
        let barrier = std::sync::Barrier::new(2);
        let writers: Vec<_> = (0..2)
            .map(|index| {
                let staging = StagedDirectory::new(&output).unwrap();
                std::fs::write(staging.path().join("audio"), [index; 128]).unwrap();
                std::fs::write(staging.path().join("descriptor"), [index]).unwrap();
                staging
            })
            .collect();
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = writers
                .into_iter()
                .map(|staging| {
                    scope.spawn(|| {
                        barrier.wait();
                        staging.publish()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        let descriptor = std::fs::read(output.join("descriptor")).unwrap();
        assert_eq!(
            std::fs::read(output.join("audio")).unwrap(),
            vec![descriptor[0]; 128]
        );
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn dangling_and_late_symlink_destinations_are_preserved() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("bundle");
        let outside = root.path().join("outside");
        std::os::unix::fs::symlink(&outside, &output).unwrap();
        assert!(StagedDirectory::new(&output).is_err());
        std::fs::remove_file(&output).unwrap();
        let staging = StagedDirectory::new(&output).unwrap();
        std::os::unix::fs::symlink(&outside, &output).unwrap();
        assert!(staging.publish().is_err());
        assert_eq!(std::fs::read_link(&output).unwrap(), outside);
        assert!(!outside.exists());
    }

    #[test]
    fn interrupted_staging_never_publishes_partial_members_and_retry_succeeds() {
        const CHILD_ROOT: &str = "SOTF_CAPTURE_PUBLICATION_TEST_ROOT";
        const CHILD_PHASE: &str = "SOTF_CAPTURE_PUBLICATION_TEST_PHASE";
        const ABRUPT_EXIT: i32 = 73;
        if let Some(root) = std::env::var_os(CHILD_ROOT) {
            let root = PathBuf::from(root);
            let phase = std::env::var(CHILD_PHASE).unwrap();
            let staging = StagedDirectory::new(&root.join("bundle")).unwrap();
            std::fs::write(staging.path().join("audio"), b"complete audio").unwrap();
            if phase != "partial" {
                std::fs::write(staging.path().join("descriptor"), b"complete descriptor").unwrap();
            }
            if phase == "published" {
                staging.publish().unwrap();
            }
            // Bypass Rust destructors, as an abruptly terminated process does.
            std::process::exit(ABRUPT_EXIT);
        }
        for phase in ["partial", "staged", "published"] {
            let root = tempfile::tempdir().unwrap();
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg("artifacts::publication::tests::interrupted_staging_never_publishes_partial_members_and_retry_succeeds")
                .arg("--nocapture")
                .env(CHILD_ROOT, root.path())
                .env(CHILD_PHASE, phase)
                .output()
                .unwrap();
            assert_eq!(
                child.status.code(),
                Some(ABRUPT_EXIT),
                "phase {phase}: {child:?}"
            );
            let output = root.path().join("bundle");
            if phase == "published" {
                assert!(StagedDirectory::new(&output).is_err());
            } else {
                assert!(!output.exists(), "phase {phase} published a partial bundle");
                let abandoned: Vec<_> = std::fs::read_dir(root.path()).unwrap().collect();
                assert_eq!(
                    abandoned.len(),
                    1,
                    "process exit must bypass staging cleanup"
                );
                let staging = StagedDirectory::new(&output).unwrap();
                std::fs::write(staging.path().join("audio"), b"complete audio").unwrap();
                std::fs::write(staging.path().join("descriptor"), b"complete descriptor").unwrap();
                staging.publish().unwrap();
                assert!(abandoned[0].as_ref().unwrap().path().is_dir());
            }
            assert_eq!(
                std::fs::read(output.join("audio")).unwrap(),
                b"complete audio"
            );
            assert_eq!(
                std::fs::read(output.join("descriptor")).unwrap(),
                b"complete descriptor"
            );
        }
    }
}
