//! `mesh-llm auth rotate-owner`: replace the owner keystore with a new
//! identity and keep the previous keystore as a backup that still unlocks.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mesh_llm_events::audit::{audit_events, emit_audit};
use mesh_llm_identity::keychain::{
    OwnerKeychainMove, OwnerKeychainMoveStart, begin_owner_keychain_move,
};
use mesh_llm_identity::{
    KEYCHAIN_SERVICE, OwnerKeypair, keystore_metadata, owner_keychain_account_for_path,
    save_keystore,
};

use super::{prompt_new_passphrase, resolve_owner_key_path};

/// The previous keystore after it was moved aside.
struct OwnerKeystoreBackup {
    path: PathBuf,
    keychain: BackupKeychain,
}

/// What happened to the previous keystore's keychain unlock secret.
enum BackupKeychain {
    /// Plaintext keystore, or no entry for its path: nothing to carry over.
    NotBound,
    /// The secret was copied to the backup's account.
    Moved(OwnerKeychainMove),
    /// No keychain backend was reachable; any secret stays under the old
    /// path's account.
    Unreachable { old_account: String },
}

pub(crate) fn run_rotate_owner(
    owner_key: Option<PathBuf>,
    no_passphrase: bool,
    force: bool,
) -> Result<()> {
    let owner_key_path = resolve_owner_key_path(owner_key)?;
    let passphrase = prompt_new_passphrase(no_passphrase)?;
    let new_keypair = OwnerKeypair::generate();

    let backup = if owner_key_path.exists() {
        Some(back_up_owner_keystore(&owner_key_path, force)?)
    } else {
        None
    };

    if let Err(error) = save_keystore(
        &owner_key_path,
        &new_keypair,
        passphrase.as_deref().map(|value| value.as_str()),
        true,
    ) {
        let error = anyhow::Error::from(error).context(format!(
            "Failed to write the new owner keystore at {}",
            owner_key_path.display()
        ));
        return Err(match backup {
            Some(backup) => backup.restore(&owner_key_path, error),
            None => error,
        });
    }

    let _ = emit_audit(audit_events::admin_action(
        Some(new_keypair.owner_id().to_string()),
        "auth_rotate_owner",
        Some("owner_keystore"),
        true,
        None,
    ));

    let mut err = mesh_llm_events::console_err();
    writeln!(
        err,
        "Rotated owner keystore at {}",
        owner_key_path.display()
    )?;
    writeln!(err, "New owner ID:    {}", new_keypair.owner_id())?;
    if let Some(backup) = backup {
        writeln!(err, "Backup:          {}", backup.path.display())?;
        report_backup_keychain(&mut err, &owner_key_path, backup)?;
    }

    Ok(())
}

/// Report where the backup's keychain secret lives, and finish moving it.
fn report_backup_keychain(
    err: &mut impl Write,
    owner_key_path: &Path,
    backup: OwnerKeystoreBackup,
) -> Result<()> {
    match backup.keychain {
        BackupKeychain::NotBound => {}
        BackupKeychain::Moved(keychain_move) => {
            writeln!(
                err,
                "Backup unlock:   OS keychain (service={KEYCHAIN_SERVICE}, account={})",
                keychain_move.destination_account()
            )?;
            // The new keystore is not bound to the keychain, so the entry
            // left under its path could only fail to unlock it.
            if let Err(error) = keychain_move.commit() {
                writeln!(
                    err,
                    "Warning: could not remove the previous keystore's keychain entry for {}: {error}",
                    owner_key_path.display()
                )?;
            }
        }
        BackupKeychain::Unreachable { old_account } => {
            writeln!(
                err,
                "Warning: no OS keychain was reachable. If the previous keystore was unlocked \
                 from the keychain, its secret is still under service={KEYCHAIN_SERVICE}, \
                 account={old_account}; store it under account={} to unlock the backup.",
                owner_keychain_account_for_path(&backup.path)
            )?;
        }
    }
    Ok(())
}

/// Move the keystore at `owner_key_path` to a timestamped backup next to it.
///
/// Keychain accounts are derived from the keystore path, so a keychain
/// unlock secret is copied to the backup's account before the file moves.
/// If the move fails, the copy is undone.
fn back_up_owner_keystore(owner_key_path: &Path, force: bool) -> Result<OwnerKeystoreBackup> {
    let timestamp = chrono::Utc::now().format("%Y%m%d%H%M%S");
    let backup_path = owner_key_path.with_file_name(format!("owner-keystore.{timestamp}.bak.json"));
    if backup_path.exists() && !force {
        bail!(
            "Backup file already exists at {} (use --force to overwrite).",
            backup_path.display()
        );
    }

    let keychain = if keystore_may_use_keychain(owner_key_path) {
        let start = begin_owner_keychain_move(owner_key_path, &backup_path).with_context(|| {
            format!(
                "Failed to copy the OS keychain entry for {} to its backup; unlock the keychain \
                 and retry so the backup stays unlockable",
                owner_key_path.display()
            )
        })?;
        match start {
            OwnerKeychainMoveStart::Started(keychain_move) => BackupKeychain::Moved(keychain_move),
            OwnerKeychainMoveStart::NoEntry => BackupKeychain::NotBound,
            OwnerKeychainMoveStart::BackendUnavailable { from_account } => {
                BackupKeychain::Unreachable {
                    old_account: from_account,
                }
            }
        }
    } else {
        BackupKeychain::NotBound
    };

    if let Err(error) = replace_with_rename(owner_key_path, &backup_path) {
        let error = anyhow::Error::from(error).context(format!(
            "Failed to move {} to {}",
            owner_key_path.display(),
            backup_path.display()
        ));
        return Err(abort_keychain_move(keychain, error));
    }

    Ok(OwnerKeystoreBackup {
        path: backup_path,
        keychain,
    })
}

impl OwnerKeystoreBackup {
    /// Put the previous keystore back after the new one could not be
    /// written, and return `error` with what happened to it.
    fn restore(self, owner_key_path: &Path, error: anyhow::Error) -> anyhow::Error {
        if owner_key_path.exists() {
            // The new keystore was published before the failure; leave
            // both files in place rather than replace it.
            return error.context(format!(
                "the previous keystore is kept at {}",
                self.path.display()
            ));
        }
        match std::fs::rename(&self.path, owner_key_path) {
            Ok(()) => abort_keychain_move(self.keychain, error).context(format!(
                "the previous keystore was restored to {}",
                owner_key_path.display()
            )),
            Err(restore_error) => error.context(format!(
                "the previous keystore is kept at {}; moving it back failed: {restore_error}",
                self.path.display()
            )),
        }
    }
}

/// Undo a keychain copy after the keystore file did not move, noting a
/// failed undo on `error`.
fn abort_keychain_move(keychain: BackupKeychain, error: anyhow::Error) -> anyhow::Error {
    let BackupKeychain::Moved(keychain_move) = keychain else {
        return error;
    };
    let account = keychain_move.destination_account().to_owned();
    match keychain_move.abort() {
        Ok(()) => error,
        Err(abort_error) => error.context(format!(
            "could not undo the keychain copy to account {account}: {abort_error}"
        )),
    }
}

/// Only an encrypted keystore can have a keychain unlock secret. A keystore
/// whose metadata cannot be read is treated as possibly encrypted.
fn keystore_may_use_keychain(path: &Path) -> bool {
    !matches!(keystore_metadata(path), Ok(info) if !info.encrypted)
}

fn replace_with_rename(from: &Path, to: &Path) -> std::io::Result<()> {
    if to.exists() {
        std::fs::remove_file(to)?;
    }
    std::fs::rename(from, to)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_identity::{
        keychain_available, keychain_delete, keychain_get, load_keystore,
        load_owner_keypair_from_keychain, owner_keychain_account_for_path,
        save_keystore_with_keychain,
    };
    use serial_test::serial;

    fn temporary_rotation_dir(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mesh-llm-rotate-owner-{label}-{}",
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn single_backup_in(dir: &Path) -> PathBuf {
        let backups: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.to_string_lossy().ends_with(".bak.json"))
            .collect();
        assert_eq!(backups.len(), 1, "expected one backup, found {backups:?}");
        backups.into_iter().next().unwrap()
    }

    #[test]
    fn failed_rotation_restores_previous_keystore() {
        let dir = temporary_rotation_dir("restore");
        let path = dir.join("owner-keystore.json");
        let previous = OwnerKeypair::generate();
        save_keystore(&path, &previous, None, false).unwrap();

        let backup = back_up_owner_keystore(&path, false).unwrap();
        assert!(!path.exists());
        let error = backup.restore(&path, anyhow::anyhow!("disk full"));

        assert!(format!("{error:#}").contains("restored"), "{error:#}");
        assert_eq!(
            load_keystore(&path, None).unwrap().owner_id(),
            previous.owner_id()
        );
        let leftover_backups = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".bak.json")
            })
            .count();
        assert_eq!(leftover_backups, 0);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rotate_owner_keeps_previous_keystore_as_backup() {
        let dir = temporary_rotation_dir("plaintext");
        let path = dir.join("owner-keystore.json");
        let previous = OwnerKeypair::generate();
        save_keystore(&path, &previous, None, false).unwrap();

        run_rotate_owner(Some(path.clone()), true, false).unwrap();

        let backup = single_backup_in(&dir);
        assert_eq!(
            load_keystore(&backup, None).unwrap().owner_id(),
            previous.owner_id()
        );
        assert_ne!(
            load_keystore(&path, None).unwrap().owner_id(),
            previous.owner_id()
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    #[serial]
    fn rotate_owner_moves_keychain_unlock_to_backup() {
        if !keychain_available() {
            eprintln!("keychain backend unavailable, skipping");
            return;
        }
        let dir = temporary_rotation_dir("keychain");
        let path = dir.join("owner-keystore.json");
        let previous = OwnerKeypair::generate();
        let original_account = match save_keystore_with_keychain(&path, &previous, false) {
            Ok(account) => account,
            // A locked collection with no one to unlock it refuses writes.
            Err(mesh_llm_identity::CryptoError::KeychainAccessDenied { reason }) => {
                eprintln!("keychain refused writes ({reason}), skipping");
                std::fs::remove_dir_all(&dir).ok();
                return;
            }
            Err(error) => panic!("keychain-bound keystore setup failed: {error}"),
        };

        run_rotate_owner(Some(path.clone()), true, false).unwrap();

        let backup = single_backup_in(&dir);
        let backup_account = owner_keychain_account_for_path(&backup);
        let unlocked = load_owner_keypair_from_keychain(&backup);
        let original_entry = keychain_get(KEYCHAIN_SERVICE, &original_account).unwrap();
        keychain_delete(KEYCHAIN_SERVICE, &backup_account).ok();
        keychain_delete(KEYCHAIN_SERVICE, &original_account).ok();
        std::fs::remove_dir_all(&dir).ok();

        let unlocked = unlocked.expect("the backup must unlock from the keychain");
        assert_eq!(unlocked.owner_id(), previous.owner_id());
        assert_eq!(
            original_entry, None,
            "the stale entry for the rotated path must be removed"
        );
    }
}
