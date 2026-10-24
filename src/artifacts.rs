//! Contained, bounded access to imported session artifacts.

// Rust guideline compliant 2026-02-21

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

mod publication;
pub(crate) use publication::StagedDirectory;

// Metadata is small even for large capture matrices. Audio allows stereo
// f32 at the existing 16-million-frame per-take limit, plus WAV headers.
pub(crate) const MAX_METADATA_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_AUDIO_BYTES: usize = 128 * 1024 * 1024;

pub(crate) fn member_name(name: &Path) -> Result<&str, String> {
    let text = name.to_str().ok_or("artifact name must be UTF-8")?;
    let mut components = name.components();
    if !autoeq::capture_handoff::portable_capture_filename(text)
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
    {
        return Err("artifact paths must be single portable relative filenames".into());
    }
    Ok(text)
}

pub(crate) fn member_path(root: &Path, name: &Path) -> Result<PathBuf, String> {
    let path = root.join(member_name(name)?);
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if !metadata.is_file() || metadata.file_type().is_symlink() => {
            Err("artifact must be a regular file, not a symlink".into())
        }
        Ok(_) => Ok(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(path),
        Err(error) => Err(format!("cannot inspect artifact: {error}")),
    }
}

pub(crate) fn open_read(path: &Path) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Refuse a final-component symlink even if it is substituted after
        // validation; NONBLOCK also prevents a substituted FIFO from hanging.
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_OPEN_REPARSE_POINT opens the link itself for rejection.
        options.custom_flags(0x0020_0000);
    }
    let file = options
        .open(path)
        .map_err(|e| format!("cannot open artifact: {e}"))?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err("artifact must be a regular file".into());
    }
    Ok(file)
}

pub(crate) fn read_file(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let file = open_read(path)?;
    if file.metadata().map_err(|e| e.to_string())?.len() > limit as u64 {
        return Err("artifact exceeds its size limit".into());
    }
    let mut data = Vec::new();
    file.take(limit as u64 + 1)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    if data.len() > limit {
        return Err("artifact exceeds its size limit".into());
    }
    Ok(data)
}

pub(crate) fn read_member(root: &Path, name: &Path, limit: usize) -> Result<Vec<u8>, String> {
    read_file(&member_path(root, name)?, limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_nonlocal_names_and_oversized_members() {
        let root = tempfile::tempdir().unwrap();
        for name in ["../outside", "/outside", "sub/file", "sub\\file"] {
            assert!(
                read_member(root.path(), Path::new(name), 16).is_err(),
                "{name}"
            );
        }
        std::fs::write(root.path().join("large"), [0; 17]).unwrap();
        assert!(read_member(root.path(), Path::new("large"), 16).is_err());
        assert_eq!(
            read_member(root.path(), Path::new("large"), 17)
                .unwrap()
                .len(),
            17
        );
    }

    #[cfg(unix)]
    #[test]
    fn refuses_symlinks_including_after_path_validation() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let name = Path::new("member");
        let path = member_path(root.path(), name).unwrap();
        std::os::unix::fs::symlink(outside.path(), &path).unwrap();
        assert!(open_read(&path).is_err());
        assert!(read_member(root.path(), name, 100).is_err());
    }
}
