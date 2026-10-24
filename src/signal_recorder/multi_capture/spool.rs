//! Bounded control-thread batches and a dedicated raw WAV writer.

// Rust guideline compliant 2026-02-21

#[cfg(test)]
use crate::limits::FILE_QUEUE_CHUNKS;
use crate::limits::{FILE_CHUNK_FRAMES, FILE_WRITER_BUFFER_BYTES};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

/// One finalized mono take awaiting publication into its session directory.
///
/// Dropping it removes the private temporary file. No public WAV is created
/// until [`Self::persist`] succeeds after the acquisition lifecycle accepts it.
#[derive(Debug)]
pub struct CapturedFile {
    path: tempfile::TempPath,
    /// Actual nominal input rate written into the WAV header.
    pub sample_rate_hz: u32,
    /// Number of complete mono frames written.
    pub samples: usize,
    /// Largest absolute sample amplitude.
    pub peak_amplitude: f32,
    /// Samples at or beyond digital full scale.
    pub clipped_samples: usize,
}

impl CapturedFile {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Publish the finalized WAV without replacing an existing destination.
    ///
    /// # Errors
    /// Returns an error if publication fails, including an existing destination
    /// or a destination on a different filesystem from the capture directory.
    pub fn persist(self, destination: &Path) -> Result<(), String> {
        self.path
            .persist_noclobber(destination)
            .map_err(|error| format!("cannot publish captured WAV: {error}"))
    }
}

trait WriteSeek: Write + Seek + Send {}
impl<T: Write + Seek + Send> WriteSeek for T {}

struct MonoWriter {
    writer: hound::WavWriter<BufWriter<Box<dyn WriteSeek>>>,
    path: tempfile::TempPath,
    rate: u32,
    samples: usize,
    peak: f32,
    clipped: usize,
}

impl MonoWriter {
    fn new(
        root: &Path,
        rate: u32,
        wrap: &mut impl FnMut(File) -> Box<dyn WriteSeek>,
    ) -> Result<Self, String> {
        let file = tempfile::Builder::new()
            .prefix(".capture-input-")
            .suffix(".wav")
            .tempfile_in(root)
            .map_err(|error| error.to_string())?;
        let (file, path) = file.into_parts();
        let writer = hound::WavWriter::new(
            BufWriter::with_capacity(FILE_WRITER_BUFFER_BYTES, wrap(file)),
            hound::WavSpec {
                channels: 1,
                sample_rate: rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .map_err(|error| format!("cannot create capture WAV: {error}"))?;
        Ok(Self {
            writer,
            path,
            rate,
            samples: 0,
            peak: 0.0,
            clipped: 0,
        })
    }

    fn write(&mut self, samples: &[f32]) -> Result<(), String> {
        for &sample in samples {
            self.writer
                .write_sample(sample)
                .map_err(|error| format!("cannot write capture WAV: {error}"))?;
            self.samples += 1;
            self.peak = self.peak.max(sample.abs());
            self.clipped += usize::from(sample.abs() >= 1.0);
        }
        Ok(())
    }

    fn finish(self) -> Result<CapturedFile, String> {
        self.writer
            .finalize()
            .map_err(|error| format!("cannot finalize capture WAV: {error}"))?;
        Ok(CapturedFile {
            path: self.path,
            sample_rate_hz: self.rate,
            samples: self.samples,
            peak_amplitude: self.peak,
            clipped_samples: self.clipped,
        })
    }
}

type Batch = (usize, Vec<f32>);
type WriterThread = JoinHandle<Result<Vec<CapturedFile>, String>>;

pub(super) struct FileSpool {
    sender: Option<mpsc::SyncSender<Batch>>,
    worker: Option<WriterThread>,
    error: Arc<Mutex<Option<String>>>,
    pending: Vec<Vec<f32>>,
    pub counts: Vec<usize>,
    controller_error: Option<String>,
    queued: usize,
    completed: Arc<AtomicUsize>,
}

impl FileSpool {
    pub fn new(root: &Path, inputs: usize, rate: u32) -> Result<Self, String> {
        Self::with_writer(root, inputs, rate, |file| Box::new(file))
    }

    fn with_writer(
        root: &Path,
        inputs: usize,
        rate: u32,
        mut wrap: impl FnMut(File) -> Box<dyn WriteSeek>,
    ) -> Result<Self, String> {
        let writers = (0..inputs)
            .map(|_| MonoWriter::new(root, rate, &mut wrap))
            .collect::<Result<Vec<_>, _>>()?;
        let (sender, receiver) =
            mpsc::sync_channel::<Batch>(crate::limits::file_queue_chunks(inputs, rate)?);
        let error = Arc::new(Mutex::new(None));
        let worker_error = Arc::clone(&error);
        let completed = Arc::new(AtomicUsize::new(0));
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::Builder::new()
            .name("capture-wav-writer".into())
            .spawn(move || {
                let result = write_batches(writers, receiver, &worker_completed);
                if let Err(message) = &result {
                    *worker_error
                        .lock()
                        .expect("capture writer error lock poisoned") = Some(message.clone());
                }
                result
            })
            .map_err(|error| format!("cannot start capture writer: {error}"))?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
            error,
            pending: (0..inputs)
                .map(|_| Vec::with_capacity(FILE_CHUNK_FRAMES))
                .collect(),
            counts: vec![0; inputs],
            controller_error: None,
            queued: 0,
            completed,
        })
    }

    pub fn check_health(&self) -> Result<(), String> {
        if let Some(error) = &self.controller_error {
            return Err(error.clone());
        }
        if let Some(error) = self
            .error
            .lock()
            .expect("capture writer error lock poisoned")
            .as_ref()
        {
            return Err(error.clone());
        }
        if self
            .worker
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished)
        {
            return Err("capture writer terminated before capture stopped".into());
        }
        Ok(())
    }

    pub fn push(&mut self, channel: usize, sample: f32) -> Result<(), String> {
        let pending = &mut self.pending[channel];
        pending.push(sample);
        if pending.len() == FILE_CHUNK_FRAMES {
            let samples = std::mem::replace(pending, Vec::with_capacity(FILE_CHUNK_FRAMES));
            let sent = self
                .sender
                .as_ref()
                .ok_or("capture writer already stopped")?
                .try_send((channel, samples))
                .map_err(|error| match error {
                    mpsc::TrySendError::Full(_) => {
                        "capture disk writer exceeded its bounded queue; disk may be too slow"
                            .to_string()
                    }
                    mpsc::TrySendError::Disconnected(_) => {
                        "capture disk writer disconnected".to_string()
                    }
                });
            if let Err(error) = sent {
                self.controller_error = Some(error.clone());
                return Err(error);
            }
            self.queued += 1;
        }
        self.counts[channel] += 1;
        Ok(())
    }

    /// Pace hardware-free producers without changing nonblocking capture behavior.
    #[cfg(any(test, feature = "test-util"))]
    pub fn settle_for_probe(&self) -> Result<(), String> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while self.completed.load(Ordering::Acquire) < self.queued {
            self.check_health()?;
            if std::time::Instant::now() >= deadline {
                return Err("file writer probe stalled".into());
            }
            std::thread::yield_now();
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Result<Vec<CapturedFile>, String> {
        let sender = self.sender.take().ok_or("capture writer already stopped")?;
        // Native streams are already released. Only finalization may wait for
        // disk I/O; the active acquisition loop always uses nonblocking sends.
        for (channel, samples) in std::mem::take(&mut self.pending).into_iter().enumerate() {
            if !samples.is_empty() && sender.send((channel, samples)).is_err() {
                break;
            }
        }
        drop(sender);
        let files = self
            .worker
            .take()
            .ok_or("capture writer already joined")?
            .join()
            .map_err(|_| "capture writer panicked".to_string())??;
        if let Some(error) = &self.controller_error {
            return Err(error.clone());
        }
        Ok(files)
    }
}

impl std::fmt::Debug for FileSpool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FileSpool")
            .field("queued", &self.queued)
            .field("completed", &self.completed.load(Ordering::Acquire))
            .field("counts", &self.counts)
            .finish_non_exhaustive()
    }
}

impl Drop for FileSpool {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn write_batches(
    mut writers: Vec<MonoWriter>,
    receiver: mpsc::Receiver<Batch>,
    completed: &AtomicUsize,
) -> Result<Vec<CapturedFile>, String> {
    for (channel, samples) in receiver {
        writers[channel].write(&samples)?;
        completed.fetch_add(1, Ordering::Release);
    }
    writers.into_iter().map(MonoWriter::finish).collect()
}

#[cfg(test)]
impl CapturedFile {
    pub(crate) fn fixture(root: &Path, rate: u32, samples: &[f32]) -> Self {
        assert!(samples.iter().all(|sample| sample.is_finite()));
        let (_, path) = tempfile::NamedTempFile::new_in(root).unwrap().into_parts();
        crate::signal_recorder::write_wav_file(&path, samples, rate, 1).unwrap();
        Self {
            path,
            sample_rate_hz: rate,
            samples: samples.len(),
            peak_amplitude: samples
                .iter()
                .fold(0.0_f32, |peak, sample| peak.max(sample.abs())),
            clipped_samples: samples.iter().filter(|sample| sample.abs() >= 1.0).count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Condvar, atomic::AtomicBool};

    #[test]
    fn long_wavs_keep_fixed_batches_and_recover_exact_samples_and_statistics() {
        let root = tempfile::tempdir().unwrap();
        let mut spool = FileSpool::new(root.path(), 2, 48000).unwrap();
        for batch in 0..128 {
            for channel in 0..2 {
                let sample = if channel == 0 {
                    -0.25
                } else if batch % 3 == 0 {
                    1.25
                } else {
                    0.75
                };
                for _ in 0..FILE_CHUNK_FRAMES {
                    spool.push(channel, sample).unwrap();
                }
            }
            spool.settle_for_probe().unwrap();
            assert!(
                spool
                    .pending
                    .iter()
                    .all(|samples| samples.capacity() == FILE_CHUNK_FRAMES)
            );
        }
        let files = spool.finish().unwrap();
        for (channel, file) in files.into_iter().enumerate() {
            assert_eq!(file.samples, 128 * FILE_CHUNK_FRAMES);
            assert_eq!(file.sample_rate_hz, 48000);
            assert_eq!(file.peak_amplitude, if channel == 0 { 0.25 } else { 1.25 });
            assert_eq!(
                file.clipped_samples,
                if channel == 0 {
                    0
                } else {
                    43 * FILE_CHUNK_FRAMES
                }
            );
            let target = root.path().join(format!("input-{channel}.wav"));
            file.persist(&target).unwrap();
            let mut reader = hound::WavReader::open(target).unwrap();
            assert_eq!(
                reader.duration(),
                u32::try_from(128 * FILE_CHUNK_FRAMES).unwrap()
            );
            for (index, sample) in reader.samples::<f32>().enumerate() {
                let expected = if channel == 0 {
                    -0.25
                } else if (index / FILE_CHUNK_FRAMES).is_multiple_of(3) {
                    1.25
                } else {
                    0.75
                };
                assert_eq!(sample.unwrap(), expected);
            }
        }
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 2);
    }

    struct FailingWriter {
        file: File,
        fail: Arc<AtomicBool>,
    }
    impl Write for FailingWriter {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            if self.fail.load(Ordering::Relaxed) {
                return Err(std::io::Error::other("injected disk full"));
            }
            self.file.write(data)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            if self.fail.load(Ordering::Relaxed) {
                return Err(std::io::Error::other("injected disk flush failure"));
            }
            self.file.flush()
        }
    }
    impl Seek for FailingWriter {
        fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
            self.file.seek(position)
        }
    }

    #[test]
    fn disk_write_and_finalization_errors_never_publish_partial_files() {
        for samples in [16, FILE_CHUNK_FRAMES * 8] {
            let root = tempfile::tempdir().unwrap();
            let fail = Arc::new(AtomicBool::new(false));
            let mut spool = FileSpool::with_writer(root.path(), 2, 48000, |file| {
                Box::new(FailingWriter {
                    file,
                    fail: Arc::clone(&fail),
                })
            })
            .unwrap();
            fail.store(true, Ordering::Relaxed);
            for _ in 0..samples {
                if spool.push(0, 0.25).is_err() {
                    break;
                }
            }
            let error = spool.finish().unwrap_err();
            assert!(error.contains("injected disk"), "{error}");
            assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
        }
    }

    struct Gate {
        blocking: AtomicBool,
        entered: AtomicBool,
        open: Mutex<bool>,
        ready: Condvar,
    }
    impl Gate {
        fn release(&self) {
            *self.open.lock().unwrap() = true;
            self.ready.notify_all();
        }
    }
    struct ReleaseOnDrop(Arc<Gate>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            self.0.release();
        }
    }
    struct GatedWriter {
        file: File,
        gate: Arc<Gate>,
    }
    impl Write for GatedWriter {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            if self.gate.blocking.load(Ordering::Relaxed) {
                self.gate.entered.store(true, Ordering::Release);
                let guard = self.gate.open.lock().unwrap();
                drop(self.gate.ready.wait_while(guard, |open| !*open).unwrap());
            }
            self.file.write(data)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.file.flush()
        }
    }
    impl Seek for GatedWriter {
        fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
            self.file.seek(position)
        }
    }

    #[test]
    fn stalled_disk_fails_at_the_bounded_queue_without_blocking_the_producer() {
        let root = tempfile::tempdir().unwrap();
        let gate = Arc::new(Gate {
            blocking: AtomicBool::new(false),
            entered: AtomicBool::new(false),
            open: Mutex::new(false),
            ready: Condvar::new(),
        });
        let mut spool = FileSpool::with_writer(root.path(), 1, 48000, |file| {
            Box::new(GatedWriter {
                file,
                gate: Arc::clone(&gate),
            })
        })
        .unwrap();
        // Declared after the spool so a failed assertion releases the worker
        // before its owner joins it during stack unwinding.
        let release = ReleaseOnDrop(Arc::clone(&gate));
        gate.blocking.store(true, Ordering::Relaxed);
        let mut error = None;
        for _ in 0..(FILE_QUEUE_CHUNKS + 4) * FILE_CHUNK_FRAMES {
            if let Err(message) = spool.push(0, 0.25) {
                error = Some(message);
                break;
            }
        }
        assert!(error.unwrap().contains("bounded queue"));
        assert_eq!(spool.pending[0].capacity(), FILE_CHUNK_FRAMES);
        drop(release);
        assert!(
            spool.finish().is_err(),
            "a queue failure must remain latched"
        );
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[test]
    fn dropping_unaccepted_files_cleans_up_and_publication_never_overwrites() {
        let root = tempfile::tempdir().unwrap();
        {
            let mut spool = FileSpool::new(root.path(), 1, 48000).unwrap();
            spool.push(0, 0.25).unwrap();
        }
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
        let target = root.path().join("existing.wav");
        std::fs::write(&target, b"sentinel").unwrap();
        let file = CapturedFile::fixture(root.path(), 48000, &[0.25; 16]);
        assert!(file.persist(&target).is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"sentinel");
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
    }
}
