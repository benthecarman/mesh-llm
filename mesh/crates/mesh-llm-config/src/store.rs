use crate::{ConfigEditor, ConfigValueSource, MeshConfig, validate_config};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use toml_edit::{ArrayOfTables, DocumentMut, Item, Table, value};

pub fn config_path(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.to_path_buf());
    }
    if let Some(path) = crate::env_overrides::config_path_override() {
        return Ok(path);
    }
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".mesh-llm").join("config.toml"))
}

pub fn load_config(override_path: Option<&Path>) -> Result<MeshConfig> {
    let path = config_path(override_path)?;
    let mut config = if path.exists() {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config {}", path.display()))?;
        parse_config_toml(&raw).with_context(|| format!("Invalid config {}", path.display()))?
    } else {
        MeshConfig::default()
    };
    crate::env_overrides::apply_env_overrides(&mut config)?;
    Ok(config)
}

pub fn parse_config_toml(raw: &str) -> Result<MeshConfig> {
    let config = parse_config_toml_structural(raw)?;
    validate_config(&config)?;
    Ok(config)
}

/// Structurally deserializes `raw` into a `MeshConfig`, performing the same
/// config-source tagging as `parse_config_toml` (e.g.
/// `runtime.lifecycle_log_parser_source`), but WITHOUT enforcing semantic
/// validation.
///
/// This exists for callers like `mesh-llm config validate` that want to
/// report semantic diagnostics themselves (so an operator sees the full
/// diagnostics list for a broken file) rather than hard-failing on the first
/// error the way `parse_config_toml` / `load_config` do for production config
/// loading, where refusing to boot on an invalid config is the correct
/// behavior.
pub fn parse_config_toml_structural(raw: &str) -> Result<MeshConfig> {
    let document: toml::Value = toml::from_str(raw).context("failed to parse config TOML")?;
    let has_lifecycle_log_parser = document
        .get("runtime")
        .and_then(toml::Value::as_table)
        .is_some_and(|runtime| runtime.contains_key("lifecycle_log_parser"));
    let mut config: MeshConfig = document.try_into().context("failed to parse config TOML")?;
    if has_lifecycle_log_parser {
        config.runtime.lifecycle_log_parser_source = ConfigValueSource::Config;
    }
    Ok(config)
}

pub fn config_to_toml(config: &MeshConfig) -> Result<String> {
    validate_config(config)?;
    toml::to_string(config).context("toml serialization failed")
}

#[derive(Clone, Debug)]
pub struct ConfigStore {
    path: PathBuf,
}

impl ConfigStore {
    pub fn open(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn default_path() -> Result<Self> {
        Ok(Self {
            path: config_path(None)?,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load(&self) -> Result<MeshConfig> {
        load_config(Some(&self.path))
    }

    pub fn save(&self, config: &MeshConfig) -> Result<()> {
        let toml_str = config_to_toml(config)?;
        let _lock = self.lock_for_write()?;
        self.write_contents(toml_str.as_bytes())
    }

    /// Reads the config file's bytes, or `None` when there is no file yet.
    /// Hand them to [`ConfigFileFingerprint::of`] and to the parser so both
    /// describe the same version of the file.
    pub fn read_contents(&self) -> Result<Option<Vec<u8>>> {
        match std::fs::read(&self.path) {
            Ok(contents) => Ok(Some(contents)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => {
                Err(error).with_context(|| format!("failed to read config {}", self.path.display()))
            }
        }
    }

    /// Saves `config` only when the file still holds what `expected` was
    /// taken from, checked while holding the write lock that the other
    /// writers here take too. A process that keeps a config in memory, such
    /// as a running host, uses this so it does not overwrite an edit someone
    /// else made to the file since it last read or wrote it.
    pub fn save_if_unchanged(
        &self,
        config: &MeshConfig,
        expected: &ConfigFileFingerprint,
    ) -> Result<ConditionalSave> {
        let toml_str = config_to_toml(config)?;
        let _lock = self.lock_for_write()?;
        let current = self.read_contents()?;
        if ConfigFileFingerprint::of(current.as_deref()) != *expected {
            return Ok(ConditionalSave::ChangedOnDisk(current));
        }
        self.write_contents(toml_str.as_bytes())?;
        Ok(ConditionalSave::Saved(ConfigFileFingerprint::of(Some(
            toml_str.as_bytes(),
        ))))
    }

    pub fn update<F>(&self, edit: F) -> Result<MeshConfig>
    where
        F: FnOnce(&mut ConfigEditor) -> Result<()>,
    {
        let _lock = self.lock_for_write()?;
        let mut editor = ConfigEditor::new(self.load()?);
        edit(&mut editor)?;
        let config = editor.into_config();
        let toml_str = config_to_toml(&config)?;
        self.write_contents(toml_str.as_bytes())?;
        Ok(config)
    }

    pub fn edit_preserving<F>(&self, edit: F) -> Result<MeshConfig>
    where
        F: FnOnce(&mut DocumentMut) -> Result<()>,
    {
        let _lock = self.lock_for_write()?;
        let mut doc = self.read_document()?;
        edit(&mut doc)?;
        let config = parse_config_toml(&doc.to_string())
            .with_context(|| format!("invalid edited config {}", self.path.display()))?;
        self.write_contents(doc.to_string().as_bytes())?;
        Ok(config)
    }

    pub fn model_refs(&self) -> Result<Vec<String>> {
        let doc = self.read_document()?;
        let Some(models) = doc.get("models").and_then(Item::as_array_of_tables) else {
            return Ok(Vec::new());
        };
        Ok(models.iter().filter_map(model_ref_from_table).collect())
    }

    pub fn add_model_ref(&self, model_ref: &str) -> Result<Vec<String>> {
        let model_ref = normalize_model_ref(model_ref)?;
        self.edit_preserving(|doc| {
            let models = ensure_models_array(doc)?;
            if !models
                .iter()
                .filter_map(model_ref_from_table)
                .any(|configured| configured == model_ref)
            {
                let mut table = Table::new();
                table["model"] = value(model_ref);
                models.push(table);
            }
            Ok(())
        })?;
        self.model_refs()
    }

    pub fn remove_model_ref(&self, model_ref: &str) -> Result<Vec<String>> {
        let model_ref = normalize_model_ref(model_ref)?;
        self.edit_preserving(|doc| {
            let Some(models) = doc.get("models").and_then(Item::as_array_of_tables) else {
                return Ok(());
            };
            let mut next = ArrayOfTables::new();
            for table in models.iter() {
                let keep = model_ref_from_table(table)
                    .map(|configured| configured != model_ref)
                    .unwrap_or(true);
                if keep {
                    next.push(table.clone());
                }
            }
            doc["models"] = Item::ArrayOfTables(next);
            Ok(())
        })?;
        self.model_refs()
    }

    fn read_document(&self) -> Result<DocumentMut> {
        if !self.path.exists() {
            return Ok(DocumentMut::new());
        }
        let raw = std::fs::read_to_string(&self.path)
            .with_context(|| format!("failed to read config {}", self.path.display()))?;
        raw.parse::<DocumentMut>()
            .with_context(|| format!("failed to parse config {}", self.path.display()))
    }

    /// Callers hold [`Self::lock_for_write`] around this.
    fn write_contents(&self, contents: &[u8]) -> Result<()> {
        crate::private_file::write_private_file(&self.path, contents)
            .with_context(|| format!("failed to write config {}", self.path.display()))
    }

    /// Takes the exclusive lock that writers of this config file hold from
    /// the read their change is based on through the write. The lock is on a
    /// `.lock` file beside the config, because each write replaces the config
    /// file itself with a new one. The lock is not reentrant: never take it
    /// twice in one thread.
    fn lock_for_write(&self) -> Result<ConfigWriteLock> {
        let lock_path = config_lock_path(&self.path);
        let file = crate::private_file::open_lock_file(&lock_path)
            .with_context(|| format!("failed to open config lock {}", lock_path.display()))?;
        match file.lock() {
            Ok(()) => {}
            // Some network and FUSE filesystems cannot lock. Writing without
            // the lock keeps config edits working there as they did before;
            // `save_if_unchanged` still compares fingerprints, it just cannot
            // exclude a writer that lands between its check and its write.
            Err(error) if locking_unsupported(&error) => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to lock config {}", self.path.display()));
            }
        }
        Ok(ConfigWriteLock { _file: file })
    }
}

/// Held while a config write is in progress; the OS releases the lock when
/// the file closes, including when the process dies.
struct ConfigWriteLock {
    _file: std::fs::File,
}

/// Whether a lock failed because the filesystem cannot lock at all, rather
/// than for a reason a retry or the operator could fix.
fn locking_unsupported(error: &std::io::Error) -> bool {
    if error.kind() == std::io::ErrorKind::Unsupported {
        return true;
    }
    #[cfg(unix)]
    if let Some(code) = error.raw_os_error() {
        return code == libc::ENOLCK || code == libc::EOPNOTSUPP;
    }
    false
}

fn config_lock_path(config_path: &Path) -> PathBuf {
    let mut name = config_path
        .file_name()
        .map(std::ffi::OsString::from)
        .unwrap_or_else(|| "config.toml".into());
    name.push(".lock");
    config_path.with_file_name(name)
}

/// Identifies one version of a config file's bytes, so a process that read
/// or wrote the file can later tell whether someone else changed it since.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigFileFingerprint {
    /// There was no file.
    Absent,
    /// SHA-256 of the file's bytes.
    Contents([u8; 32]),
}

impl ConfigFileFingerprint {
    pub fn of(contents: Option<&[u8]>) -> Self {
        use sha2::{Digest, Sha256};
        match contents {
            None => Self::Absent,
            Some(contents) => Self::Contents(Sha256::digest(contents).into()),
        }
    }
}

/// The outcome of [`ConfigStore::save_if_unchanged`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConditionalSave {
    /// The config was written; this fingerprints what is on disk now.
    Saved(ConfigFileFingerprint),
    /// Someone changed the file since the expected version, so nothing was
    /// written. Holds what the file contained when it was compared, or
    /// `None` when it had been removed, so the caller can take that version
    /// in without reading the file again outside the lock.
    ChangedOnDisk(Option<Vec<u8>>),
}

fn ensure_models_array(doc: &mut DocumentMut) -> Result<&mut ArrayOfTables> {
    if !doc.as_table().contains_key("models") {
        doc["models"] = Item::ArrayOfTables(ArrayOfTables::new());
    }
    doc["models"]
        .as_array_of_tables_mut()
        .ok_or_else(|| anyhow::anyhow!("config key `models` is not a TOML array of tables"))
}

fn model_ref_from_table(table: &Table) -> Option<String> {
    table
        .get("model")
        .and_then(Item::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn normalize_model_ref(model_ref: &str) -> Result<&str> {
    let model_ref = model_ref.trim();
    if model_ref.is_empty() {
        bail!("model ref cannot be empty");
    }
    Ok(model_ref)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_model(model: &str) -> MeshConfig {
        let mut config = MeshConfig::default();
        config.models.push(crate::ModelConfigEntry {
            model: model.to_string(),
            ..Default::default()
        });
        config
    }

    #[test]
    fn save_if_unchanged_writes_over_the_version_it_expected() {
        let dir = tempfile::tempdir().unwrap();
        let store = ConfigStore::open(dir.path().join("config.toml"));

        let saved = store
            .save_if_unchanged(&config_with_model("a.gguf"), &ConfigFileFingerprint::Absent)
            .unwrap();

        let ConditionalSave::Saved(fingerprint) = saved else {
            panic!("expected a save, got {saved:?}");
        };
        let contents = store.read_contents().unwrap();
        assert_eq!(fingerprint, ConfigFileFingerprint::of(contents.as_deref()));
        assert_eq!(
            store
                .save_if_unchanged(&config_with_model("b.gguf"), &fingerprint)
                .unwrap(),
            ConditionalSave::Saved(ConfigFileFingerprint::of(
                store.read_contents().unwrap().as_deref()
            ))
        );
        assert_eq!(store.model_refs().unwrap(), ["b.gguf"]);
    }

    #[test]
    fn save_if_unchanged_leaves_a_file_someone_else_changed() {
        let dir = tempfile::tempdir().unwrap();
        let store = ConfigStore::open(dir.path().join("config.toml"));
        let before = ConfigFileFingerprint::of(store.read_contents().unwrap().as_deref());
        store.add_model_ref("cli.gguf").unwrap();
        let edited = std::fs::read(store.path()).unwrap();

        let outcome = store
            .save_if_unchanged(&config_with_model("host.gguf"), &before)
            .unwrap();

        assert_eq!(
            outcome,
            ConditionalSave::ChangedOnDisk(Some(edited.clone()))
        );
        assert_eq!(std::fs::read(store.path()).unwrap(), edited);
    }

    #[test]
    fn only_a_filesystem_that_cannot_lock_skips_the_lock() {
        assert!(locking_unsupported(&std::io::Error::from(
            std::io::ErrorKind::Unsupported
        )));
        #[cfg(unix)]
        assert!(locking_unsupported(&std::io::Error::from_raw_os_error(
            libc::ENOLCK
        )));
        assert!(!locking_unsupported(&std::io::Error::from(
            std::io::ErrorKind::PermissionDenied
        )));
        assert!(!locking_unsupported(&std::io::Error::from(
            std::io::ErrorKind::Interrupted
        )));
    }

    #[test]
    fn writers_hold_the_lock_file_beside_the_config() {
        let dir = tempfile::tempdir().unwrap();
        let store = ConfigStore::open(dir.path().join("config.toml"));
        let lock_path = dir.path().join("config.toml.lock");

        let guard = store.lock_for_write().unwrap();
        let other = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        assert!(
            matches!(other.try_lock(), Err(std::fs::TryLockError::WouldBlock)),
            "a second writer must wait while the lock is held"
        );
        drop(guard);
        other.try_lock().expect("the lock is free once released");
    }
}
