//! Owner-only placement for secret key files: private parent directories and
//! owner-only key files, for the node key and the owner keystore. The owner
//! keystore is often provisioned by other tools, so its file check accepts
//! symlinks and owner-chosen group access. `mesh-llm-config` keeps its own
//! private-file checks for configuration; identity cannot depend on it.

use std::path::Path;

use crate::CryptoError;

/// Create any missing ancestors of `dir` with mode 0700 on Unix.
///
/// Existing directories keep their mode: a shared parent such as a home
/// directory is not tightened behind the user's back. `what` names the
/// secret in error messages, for example `"node key"`.
pub(crate) fn ensure_private_dir(dir: &Path, what: &str) -> Result<(), CryptoError> {
    let mut missing = Vec::new();
    let mut current = dir;
    loop {
        // A relative path such as `keys/node.key` bottoms out at the empty
        // path, which is the current directory and already exists.
        if current.as_os_str().is_empty() {
            break;
        }
        match std::fs::metadata(current) {
            Ok(metadata) => {
                if !metadata.is_dir() {
                    return Err(not_a_directory(current, what));
                }
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                missing.push(current);
                current = current.parent().ok_or_else(|| {
                    CryptoError::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("cannot find parent directory for {}", dir.display()),
                    ))
                })?;
            }
            Err(error) => return Err(error.into()),
        }
    }

    for path in missing.into_iter().rev() {
        match create_private_dir(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if !std::fs::metadata(path)?.is_dir() {
                    return Err(not_a_directory(path, what));
                }
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn not_a_directory(path: &Path, what: &str) -> CryptoError {
    CryptoError::Io(std::io::Error::new(
        std::io::ErrorKind::NotADirectory,
        format!("{what} parent {} is not a directory", path.display()),
    ))
}

#[cfg(unix)]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;

    std::fs::DirBuilder::new().mode(0o700).create(path)
}

#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir(path)
}

/// Require `path` to be a regular file (not a symlink) and, on Unix, reset
/// group or other permission bits to owner-only 0600.
#[cfg(unix)]
pub(crate) fn ensure_private_file(path: &Path, what: &str) -> Result<(), CryptoError> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(not_a_regular_file(path, what));
    }
    let mut perms = metadata.permissions();
    if perms.mode() & 0o077 != 0 {
        perms.set_mode(0o600);
        std::fs::set_permissions(path, perms)?;
    }
    Ok(())
}

/// Require `path` to be a regular file (not a symlink).
#[cfg(not(unix))]
pub(crate) fn ensure_private_file(path: &Path, what: &str) -> Result<(), CryptoError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(not_a_regular_file(path, what));
    }
    Ok(())
}

/// Check a secret file that may have been provisioned by someone else,
/// such as a secrets manager or a service account.
///
/// Unlike [`ensure_private_file`], `path` may be a symlink; its target must
/// be a regular file. Group or other permission bits are reset to 0600 when
/// this process may change the mode. When it may not (the file belongs to
/// another user, or the filesystem is read-only), group access is accepted
/// as the owner's choice, but a file readable by all users is refused.
#[cfg(unix)]
pub(crate) fn ensure_private_provisioned_file(path: &Path, what: &str) -> Result<(), CryptoError> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(not_a_regular_file(path, what));
    }
    let mode = metadata.permissions().mode();
    if mode & 0o077 == 0 {
        return Ok(());
    }
    match std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
        Ok(()) => Ok(()),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ReadOnlyFilesystem
            ) =>
        {
            if mode & 0o007 != 0 {
                Err(CryptoError::InvalidKeyMaterial {
                    reason: format!(
                        "{what} {} is readable by all users (mode {:03o}) and this process \
                         cannot change its mode: {error}; restrict it to its owner",
                        path.display(),
                        mode & 0o777
                    ),
                })
            } else {
                Ok(())
            }
        }
        Err(error) => Err(error.into()),
    }
}

/// Require `path`, or the target of a symlink at `path`, to be a regular file.
#[cfg(not(unix))]
pub(crate) fn ensure_private_provisioned_file(path: &Path, what: &str) -> Result<(), CryptoError> {
    if !std::fs::metadata(path)?.is_file() {
        return Err(not_a_regular_file(path, what));
    }
    Ok(())
}

fn not_a_regular_file(path: &Path, what: &str) -> CryptoError {
    CryptoError::InvalidKeyMaterial {
        reason: format!("{what} path {} is not a regular file", path.display()),
    }
}
