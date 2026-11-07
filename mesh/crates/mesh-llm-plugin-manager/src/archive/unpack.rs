//! Unpacking a plugin archive into its staging directory, within limits on
//! how much it may write. A plugin archive comes from a release anyone may
//! publish, and a small compressed archive can expand without bound; the
//! limits refuse it before it fills the disk or memory.

use std::{
    cell::Cell,
    fs,
    io::{self, Read},
    path::Path,
    rc::Rc,
};

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;

use crate::target::ArchiveExt;

/// How much an archive may unpack to.
#[derive(Debug, Clone, Copy)]
pub(super) struct UnpackLimits {
    /// The total size of the files the archive holds, counted as unpacked.
    pub max_bytes: u64,
    /// The number of entries: files, directories, and links.
    pub max_entries: u64,
    /// The tar records between one entry's data and the next entry: block
    /// padding, pax and long-name records, and the header. tar reads them
    /// into memory whole, so this bounds what one entry can make it hold.
    pub max_record_bytes: u64,
}

impl UnpackLimits {
    /// The limits for a plugin package: an executable, a manifest, and web UI
    /// assets. All are far above any real plugin; the record allowance leaves
    /// room for the extended attributes some tar writers record.
    pub const PLUGIN: Self = Self {
        max_bytes: 1024 * 1024 * 1024,
        max_entries: 10_000,
        max_record_bytes: 1024 * 1024,
    };
}

/// A tar block: the unit headers and data padding come in.
const TAR_BLOCK_BYTES: u64 = 512;

/// Unpack `archive_path` into `destination`, refusing an archive over
/// `limits`. What was written before the refusal stays in `destination`,
/// which is the caller's staging directory.
pub(super) fn unpack_archive(
    archive_path: &Path,
    archive_ext: ArchiveExt,
    destination: &Path,
    limits: UnpackLimits,
) -> Result<()> {
    match archive_ext {
        ArchiveExt::TarGz => unpack_tar_gz(archive_path, destination, limits),
        ArchiveExt::Zip => unpack_zip(archive_path, destination, limits),
    }
    .with_context(|| format!("extract plugin archive {}", archive_path.display()))
}

/// What an archive has used of its limits so far.
struct UnpackBudget {
    limits: UnpackLimits,
    bytes: u64,
    entries: u64,
}

impl UnpackBudget {
    fn new(limits: UnpackLimits) -> Self {
        Self {
            limits,
            bytes: 0,
            entries: 0,
        }
    }

    fn admit_entries(&mut self, count: u64) -> Result<()> {
        self.entries = self.entries.saturating_add(count);
        if self.entries > self.limits.max_entries {
            bail!(
                "plugin archive has more than {} entries",
                self.limits.max_entries
            );
        }
        Ok(())
    }

    fn admit_bytes(&mut self, count: u64) -> Result<()> {
        self.bytes = self.bytes.saturating_add(count);
        if self.bytes > self.limits.max_bytes {
            bail!(
                "plugin archive unpacks to more than {} bytes",
                self.limits.max_bytes
            );
        }
        Ok(())
    }

    fn remaining_bytes(&self) -> u64 {
        self.limits.max_bytes.saturating_sub(self.bytes)
    }
}

/// How far into the decompressed tar stream tar may read. tar reads an
/// entry's pax and long-name records whole, before it yields the entry, so a
/// cap on the whole stream would let one record take up to the whole cap in
/// memory. Instead, each yielded entry opens the window only as far as its
/// own stored data and the records that lead to the next entry.
struct TarReadWindow {
    read: Cell<u64>,
    limit: Cell<u64>,
    max_record_bytes: u64,
}

impl TarReadWindow {
    fn new(max_record_bytes: u64) -> Self {
        Self {
            read: Cell::new(0),
            limit: Cell::new(max_record_bytes),
            max_record_bytes,
        }
    }

    /// Let tar read the `stored` data bytes of the entry it just yielded and
    /// then the records up to and including the next entry's header.
    fn open_past(&self, stored: u64) {
        let limit = self
            .read
            .get()
            .saturating_add(stored)
            .saturating_add(TAR_BLOCK_BYTES)
            .saturating_add(self.max_record_bytes);
        self.limit.set(limit);
    }
}

/// The decompressed tar stream, failing a read that would pass its window.
struct WindowedReader<R> {
    inner: R,
    window: Rc<TarReadWindow>,
}

impl<R: Read> Read for WindowedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read(buf)?;
        let read = self.window.read.get().saturating_add(count as u64);
        self.window.read.set(read);
        if read > self.window.limit.get() {
            return Err(io::Error::other(format!(
                "plugin archive has tar records over {} bytes between entries",
                self.window.max_record_bytes
            )));
        }
        Ok(count)
    }
}

/// The data bytes `entry` occupies in the stream. For a GNU sparse file that
/// is the header's size field; `Entry::size` is what the holes unpack to.
fn tar_stored_bytes<R: Read>(entry: &tar::Entry<'_, R>) -> Result<u64> {
    if entry.header().entry_type().is_gnu_sparse() {
        return entry
            .header()
            .entry_size()
            .context("read plugin tar sparse entry size");
    }
    Ok(entry.size())
}

/// Unpack as `tar::Archive::unpack` does, directories last so their modes do
/// not block their contents, while counting each entry and its size against
/// the budget before it is written. An entry's size is what it unpacks to,
/// which for a sparse file is more than the archive holds.
fn unpack_tar_gz(archive_path: &Path, destination: &Path, limits: UnpackLimits) -> Result<()> {
    let file = fs::File::open(archive_path)
        .with_context(|| format!("open plugin archive {}", archive_path.display()))?;
    let window = Rc::new(TarReadWindow::new(limits.max_record_bytes));
    let mut archive = tar::Archive::new(WindowedReader {
        inner: GzDecoder::new(file),
        window: Rc::clone(&window),
    });
    fs::create_dir_all(destination)
        .with_context(|| format!("create extract directory {}", destination.display()))?;
    let destination = destination
        .canonicalize()
        .unwrap_or_else(|_| destination.to_path_buf());

    let mut budget = UnpackBudget::new(limits);
    let mut directories = Vec::new();
    for entry in archive.entries().context("read plugin tar archive")? {
        let mut entry = entry.context("read plugin tar entry")?;
        budget.admit_entries(1)?;
        budget.admit_bytes(entry.size())?;
        window.open_past(tar_stored_bytes(&entry)?);
        if entry.header().entry_type() == tar::EntryType::Directory {
            directories.push(entry);
        } else {
            entry
                .unpack_in(&destination)
                .context("unpack plugin tar entry")?;
        }
    }
    directories.sort_by(|a, b| b.path_bytes().cmp(&a.path_bytes()));
    for mut directory in directories {
        directory
            .unpack_in(&destination)
            .context("unpack plugin tar directory")?;
    }
    Ok(())
}

/// Unpack a zip, counting what each file actually decompresses to rather
/// than the size its header declares.
fn unpack_zip(archive_path: &Path, destination: &Path, limits: UnpackLimits) -> Result<()> {
    let file = fs::File::open(archive_path)
        .with_context(|| format!("open plugin archive {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("read plugin zip archive {}", archive_path.display()))?;
    let mut budget = UnpackBudget::new(limits);
    budget.admit_entries(archive.len() as u64)?;
    for index in 0..archive.len() {
        let mut file = archive.by_index(index)?;
        let Some(enclosed) = file.enclosed_name() else {
            bail!(
                "zip archive contains unsafe path: {}",
                String::from_utf8_lossy(file.name_raw())
            );
        };
        let output_path = destination.join(enclosed);
        if file.is_dir() {
            fs::create_dir_all(&output_path)
                .with_context(|| format!("create zip directory {}", output_path.display()))?;
            continue;
        }
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create zip parent directory {}", parent.display()))?;
        }
        let mut output = fs::File::create(&output_path)
            .with_context(|| format!("create zip output {}", output_path.display()))?;
        let allowed = budget.remaining_bytes().saturating_add(1);
        let written = io::copy(&mut (&mut file).take(allowed), &mut output)
            .with_context(|| format!("write zip output {}", output_path.display()))?;
        budget.admit_bytes(written)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::{Compression, write::GzEncoder};
    use tempfile::TempDir;

    use super::*;

    const SMALL: UnpackLimits = UnpackLimits {
        max_bytes: 1024,
        max_entries: 4,
        max_record_bytes: 4096,
    };

    fn write_tar_gz(path: &Path, files: &[(&str, &[u8])]) {
        let encoder = GzEncoder::new(fs::File::create(path).unwrap(), Compression::default());
        let mut archive = tar::Builder::new(encoder);
        for (name, contents) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive.append_data(&mut header, name, *contents).unwrap();
        }
        archive.into_inner().unwrap().finish().unwrap();
    }

    fn write_zip(path: &Path, files: &[(&str, &[u8])]) {
        let mut archive = zip::ZipWriter::new(fs::File::create(path).unwrap());
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, contents) in files {
            archive.start_file(*name, options).unwrap();
            archive.write_all(contents).unwrap();
        }
        archive.finish().unwrap();
    }

    /// Unpack `files` packed as `ext` under `limits`, into a fresh directory.
    fn unpack_files(ext: ArchiveExt, files: &[(&str, &[u8])], limits: UnpackLimits) -> Result<()> {
        let temp = TempDir::new().unwrap();
        let archive = temp.path().join("demo.archive");
        match ext {
            ArchiveExt::TarGz => write_tar_gz(&archive, files),
            ArchiveExt::Zip => write_zip(&archive, files),
        }
        unpack_archive(&archive, ext, &temp.path().join("out"), limits)
    }

    #[test]
    fn an_archive_within_its_limits_unpacks() {
        for ext in [ArchiveExt::TarGz, ArchiveExt::Zip] {
            let temp = TempDir::new().unwrap();
            let archive = temp.path().join("demo.archive");
            let files: &[(&str, &[u8])] =
                &[("demo/plugin.toml", b"name"), ("demo/demo", &[1; 1000])];
            match ext {
                ArchiveExt::TarGz => write_tar_gz(&archive, files),
                ArchiveExt::Zip => write_zip(&archive, files),
            }
            let out = temp.path().join("out");
            unpack_archive(&archive, ext, &out, SMALL).unwrap();
            assert_eq!(fs::read(out.join("demo/demo")).unwrap(), vec![1; 1000]);
        }
    }

    /// A few kilobytes of zeros compress to almost nothing; what counts is
    /// what they unpack to.
    #[test]
    fn an_archive_that_unpacks_past_the_byte_limit_is_refused() {
        for ext in [ArchiveExt::TarGz, ArchiveExt::Zip] {
            let error = unpack_files(
                ext,
                &[("demo/plugin.toml", b"name"), ("demo/demo", &[0; 4096])],
                SMALL,
            )
            .unwrap_err();
            assert!(
                format!("{error:#}").contains("unpacks to more than 1024 bytes"),
                "{ext:?}: {error:#}"
            );
        }
    }

    #[test]
    fn an_archive_with_too_many_entries_is_refused() {
        let files: Vec<(String, Vec<u8>)> = (0..5)
            .map(|index| (format!("demo/file-{index}"), Vec::new()))
            .collect();
        let files: Vec<(&str, &[u8])> = files
            .iter()
            .map(|(name, contents)| (name.as_str(), contents.as_slice()))
            .collect();
        for ext in [ArchiveExt::TarGz, ArchiveExt::Zip] {
            let error = unpack_files(ext, &files, SMALL).unwrap_err();
            assert!(
                format!("{error:#}").contains("more than 4 entries"),
                "{ext:?}: {error:#}"
            );
        }
    }

    /// Write `value` as a NUL-terminated octal tar number field.
    fn octal_field(field: &mut [u8], value: u64) {
        let text = format!("{value:0width$o}\0", width = field.len() - 1);
        field.copy_from_slice(text.as_bytes());
    }

    /// A GNU sparse entry holds no data but unpacks to `real_size` bytes of
    /// zeros. Counting the bytes the archive carries would admit it.
    #[test]
    fn a_sparse_entry_counts_the_size_it_unpacks_to() {
        let temp = TempDir::new().unwrap();
        let archive_path = temp.path().join("sparse.tar.gz");
        let real_size = 1024 * 1024 * 1024 * 2;
        let mut header = tar::Header::new_gnu();
        header.set_path("demo/hole").unwrap();
        header.set_entry_type(tar::EntryType::GNUSparse);
        header.set_mode(0o644);
        header.set_size(0);
        let gnu = header.as_gnu_mut().unwrap();
        octal_field(&mut gnu.realsize, real_size);
        octal_field(&mut gnu.sparse[0].offset, real_size);
        octal_field(&mut gnu.sparse[0].numbytes, 0);
        header.set_cksum();
        let encoder = GzEncoder::new(
            fs::File::create(&archive_path).unwrap(),
            Compression::default(),
        );
        let mut archive = tar::Builder::new(encoder);
        archive.append(&header, io::empty()).unwrap();
        archive.into_inner().unwrap().finish().unwrap();

        let error = unpack_archive(
            &archive_path,
            ArchiveExt::TarGz,
            &temp.path().join("out"),
            SMALL,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("unpacks to more than 1024 bytes"),
            "{error:#}"
        );
        assert!(!temp.path().join("out/demo/hole").exists());
    }

    /// Write a tar.gz of a 700-byte `demo/plugin.toml`, then a pax record of
    /// `pax_bytes`, then an empty `demo/demo`.
    fn write_tar_gz_with_pax_record(path: &Path, pax_bytes: usize) {
        let encoder = GzEncoder::new(fs::File::create(path).unwrap(), Compression::default());
        let mut archive = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(700);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(&mut header, "demo/plugin.toml", &[b'n'; 700][..])
            .unwrap();
        // One pax record, `<length> comment=<value>\n`, whose length counts
        // its own four digits.
        assert!((1000..10_000).contains(&pax_bytes));
        let record = format!("{pax_bytes} comment={}\n", "x".repeat(pax_bytes - 14)).into_bytes();
        assert_eq!(record.len(), pax_bytes);
        let mut pax = tar::Header::new_ustar();
        pax.set_entry_type(tar::EntryType::XHeader);
        pax.set_size(record.len() as u64);
        pax.set_mode(0o644);
        pax.set_cksum();
        archive
            .append_data(&mut pax, "demo/pax", record.as_slice())
            .unwrap();
        let mut header = tar::Header::new_gnu();
        header.set_size(0);
        header.set_mode(0o755);
        header.set_cksum();
        archive
            .append_data(&mut header, "demo/demo", io::empty())
            .unwrap();
        archive.into_inner().unwrap().finish().unwrap();
    }

    /// tar reads a pax record into memory whole before it yields the entry
    /// the record describes, and the record has no entry size to count. The
    /// window refuses one over the record allowance even after earlier
    /// entries, whose data does not widen what a later record may take.
    #[test]
    fn a_tar_record_over_the_record_allowance_is_refused() {
        let temp = TempDir::new().unwrap();
        let archive_path = temp.path().join("pax.tar.gz");
        write_tar_gz_with_pax_record(&archive_path, 8 * 1024);

        let error = unpack_archive(
            &archive_path,
            ArchiveExt::TarGz,
            &temp.path().join("out"),
            SMALL,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("tar records over 4096 bytes between entries"),
            "{error:#}"
        );
    }

    /// The window opens for each entry's data and the records after it, so an
    /// archive of ordinary entries and long names unpacks in full.
    #[test]
    fn tar_records_within_the_allowance_unpack() {
        let temp = TempDir::new().unwrap();
        let archive_path = temp.path().join("names.tar.gz");
        write_tar_gz_with_pax_record(&archive_path, 1024);
        let out = temp.path().join("out");
        unpack_archive(&archive_path, ArchiveExt::TarGz, &out, SMALL).unwrap();
        assert_eq!(fs::read(out.join("demo/plugin.toml")).unwrap().len(), 700);
        assert!(out.join("demo/demo").is_file());

        let long_name = format!("demo/{}name", "directory/".repeat(30));
        write_tar_gz(&archive_path, &[(long_name.as_str(), b"name")]);
        let out = temp.path().join("long");
        unpack_archive(&archive_path, ArchiveExt::TarGz, &out, SMALL).unwrap();
        assert_eq!(fs::read(out.join(&long_name)).unwrap(), b"name");
    }
}
