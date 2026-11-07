use std::io::Write;
use std::path::{Path, PathBuf};

use argon2::{Argon2, Params};
use chacha20poly1305::aead::{Aead, Generate};
use chacha20poly1305::{ChaCha20Poly1305, KeyInit};
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use super::{CryptoError, OwnerKeypair, owner_id_from_verifying_key};
use crate::private_file::{ensure_private_dir, ensure_private_provisioned_file};

const KEYSTORE_VERSION: u32 = 1;

// Fixed Argon2id KDF parameters used for new keystores.
// These match argon2 0.5.x defaults and are stored explicitly in the JSON
// so that future crate upgrades cannot silently change key derivation.
const ARGON2_M_COST: u32 = 19456; // 19 MiB
const ARGON2_T_COST: u32 = 2;
const ARGON2_P_COST: u32 = 1;

// Ceilings on the Argon2id costs read back from a keystore file. The file
// chooses its own costs, so without a limit a damaged or planted keystore
// can make an unlock allocate gigabytes or run for hours. The limits leave
// room for keystores written with much stronger costs than the defaults.
const MAX_ARGON2_M_COST: u32 = 256 * 1024; // 256 MiB
const MAX_ARGON2_T_COST: u32 = 10;
const MAX_ARGON2_P_COST: u32 = 16;

/// Names the owner keystore in file-placement errors.
const KEYSTORE_FILE_LABEL: &str = "owner keystore";

/// On-disk keystore format (JSON).
#[derive(Serialize, Deserialize)]
struct KeystoreV1 {
    version: u32,
    owner_id: String,
    created_at: String,
    encrypted: bool,

    // Present when encrypted == false
    signing_secret_key: Option<String>,
    signing_public_key: Option<String>,
    encryption_secret_key: Option<String>,
    encryption_public_key: Option<String>,

    // Present when encrypted == true
    kdf: Option<String>,
    argon2_salt: Option<String>,
    // Explicit KDF parameters stored for forward compatibility (new keystores only).
    argon2_m_cost: Option<u32>,
    argon2_t_cost: Option<u32>,
    argon2_p_cost: Option<u32>,
    nonce: Option<String>,
    ciphertext: Option<String>,
}

/// Plaintext inner payload that gets encrypted when passphrase is set.
#[derive(Serialize, Deserialize)]
struct SecretPayload {
    signing_secret_key: String,
    encryption_secret_key: String,
}

/// Return the default keystore path: `~/.mesh-llm/owner-keystore.json`.
pub fn default_keystore_path() -> Result<PathBuf, CryptoError> {
    let home = dirs::home_dir().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot determine home directory",
        ))
    })?;
    Ok(home.join(".mesh-llm").join("owner-keystore.json"))
}

/// Check if a keystore file exists at the given path.
pub fn keystore_exists(path: &Path) -> bool {
    path.exists()
}

/// Save an owner keypair to disk.
///
/// If `passphrase` is `Some`, the secret keys are encrypted with Argon2id + ChaCha20Poly1305.
/// File permissions are set to 0600 on Unix, and missing parent directories
/// are created 0700.
///
/// Returns `CryptoError::KeystoreAlreadyExists` if a keystore already exists at `path`
/// and `overwrite` is `false`. Without `overwrite` the file is published
/// without replacing anything, so when two writers race only one succeeds.
pub fn save_keystore(
    path: &Path,
    keypair: &OwnerKeypair,
    passphrase: Option<&str>,
    overwrite: bool,
) -> Result<(), CryptoError> {
    // Fail fast before the KDF; publication below is the authoritative check.
    if path.exists() && !overwrite {
        return Err(keystore_already_exists(path));
    }

    if let Some(parent) = path.parent() {
        ensure_private_dir(parent, KEYSTORE_FILE_LABEL)?;
    }

    let keystore = if let Some(pass) = passphrase {
        build_encrypted_keystore(keypair, pass)?
    } else {
        build_plaintext_keystore(keypair)
    };

    let json = serde_json::to_string_pretty(&keystore)?;
    if overwrite {
        write_keystore_bytes_atomically(path, json.as_bytes())
    } else {
        publish_keystore_bytes_without_replacing(path, json.as_bytes())
    }
}

/// Load an owner keypair from disk.
///
/// If the keystore is encrypted, `passphrase` must be provided.
pub fn load_keystore(path: &Path, passphrase: Option<&str>) -> Result<OwnerKeypair, CryptoError> {
    let ks = read_keystore_file(path)?;

    let keypair = if ks.encrypted {
        decrypt_keystore(&ks, passphrase)?
    } else {
        plaintext_keystore(&ks)?
    };

    // Validate stored metadata against the derived keypair to detect tampering.
    validate_keystore_metadata(&ks, &keypair)?;

    Ok(keypair)
}

/// Read keystore metadata without decrypting secret keys.
pub fn keystore_metadata(path: &Path) -> Result<KeystoreInfo, CryptoError> {
    let ks = read_keystore_file(path)?;

    let signing_public_key =
        ks.signing_public_key
            .clone()
            .ok_or_else(|| CryptoError::InvalidKeyMaterial {
                reason: "missing signing_public_key in keystore metadata".into(),
            })?;
    let signing_public_key_bytes: [u8; 32] = hex::decode(&signing_public_key)
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("bad signing public key hex: {e}"),
        })?
        .try_into()
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: "signing public key must be 32 bytes".into(),
        })?;
    let signing_public_key = ed25519_dalek::VerifyingKey::from_bytes(&signing_public_key_bytes)
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("invalid signing public key: {e}"),
        })?;
    let verified_owner_id = owner_id_from_verifying_key(&signing_public_key);
    if ks.owner_id != verified_owner_id {
        return Err(CryptoError::VerificationFailed {
            reason: "owner_id does not match signing public key".into(),
        });
    }

    let encryption_public_key =
        ks.encryption_public_key
            .clone()
            .ok_or_else(|| CryptoError::InvalidKeyMaterial {
                reason: "missing encryption_public_key in keystore metadata".into(),
            })?;
    let encryption_public_key_bytes: [u8; 32] = hex::decode(&encryption_public_key)
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("bad encryption public key hex: {e}"),
        })?
        .try_into()
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: "encryption public key must be 32 bytes".into(),
        })?;

    Ok(KeystoreInfo {
        owner_id: verified_owner_id,
        created_at: ks.created_at,
        encrypted: ks.encrypted,
        signing_public_key: Some(hex::encode(signing_public_key.as_bytes())),
        encryption_public_key: Some(hex::encode(encryption_public_key_bytes)),
    })
}

/// Public metadata about a keystore (no secrets).
pub struct KeystoreInfo {
    pub owner_id: String,
    pub created_at: String,
    pub encrypted: bool,
    pub signing_public_key: Option<String>,
    pub encryption_public_key: Option<String>,
}

// ── Internal helpers ────────────────────────────────────────────────

/// Read and parse a keystore file after checking that it, or its symlink
/// target, is a private regular file. Group or other permission bits are
/// reset to 0600 where this process may change them.
fn read_keystore_file(path: &Path) -> Result<KeystoreV1, CryptoError> {
    if !path.exists() {
        return Err(CryptoError::KeystoreNotFound {
            path: path.display().to_string(),
        });
    }
    ensure_private_provisioned_file(path, KEYSTORE_FILE_LABEL)?;

    let raw = std::fs::read_to_string(path)?;
    let ks: KeystoreV1 = serde_json::from_str(&raw)?;

    if ks.version != KEYSTORE_VERSION {
        return Err(CryptoError::UnsupportedVersion {
            version: ks.version,
        });
    }
    Ok(ks)
}

fn keystore_already_exists(path: &Path) -> CryptoError {
    CryptoError::KeystoreAlreadyExists {
        path: path.display().to_string(),
    }
}

fn build_plaintext_keystore(keypair: &OwnerKeypair) -> KeystoreV1 {
    KeystoreV1 {
        version: KEYSTORE_VERSION,
        owner_id: keypair.owner_id(),
        created_at: chrono::Utc::now().to_rfc3339(),
        encrypted: false,
        signing_secret_key: Some(hex::encode(keypair.signing_bytes())),
        signing_public_key: Some(hex::encode(keypair.verifying_key().as_bytes())),
        encryption_secret_key: Some(hex::encode(keypair.encryption_bytes())),
        encryption_public_key: Some(hex::encode(keypair.encryption_public_key().as_bytes())),
        kdf: None,
        argon2_salt: None,
        argon2_m_cost: None,
        argon2_t_cost: None,
        argon2_p_cost: None,
        nonce: None,
        ciphertext: None,
    }
}

fn build_encrypted_keystore(
    keypair: &OwnerKeypair,
    passphrase: &str,
) -> Result<KeystoreV1, CryptoError> {
    let salt: [u8; 16] = rand::random();

    // Derive a 32-byte symmetric key from the passphrase using explicit params.
    let sym_key = derive_key(
        passphrase,
        &salt,
        ARGON2_M_COST,
        ARGON2_T_COST,
        ARGON2_P_COST,
    )?;

    // Encrypt the secret keys.
    let payload = SecretPayload {
        signing_secret_key: hex::encode(keypair.signing_bytes()),
        encryption_secret_key: hex::encode(keypair.encryption_bytes()),
    };
    let plaintext = serde_json::to_vec(&payload)?;

    let key = chacha20poly1305::Key::from(*sym_key);
    let cipher = ChaCha20Poly1305::new(&key);
    let nonce = chacha20poly1305::Nonce::generate();
    let ct = cipher
        .encrypt(&nonce, plaintext.as_ref())
        .map_err(|_| CryptoError::DecryptionFailed)?;

    Ok(KeystoreV1 {
        version: KEYSTORE_VERSION,
        owner_id: keypair.owner_id(),
        created_at: chrono::Utc::now().to_rfc3339(),
        encrypted: true,
        signing_secret_key: None,
        signing_public_key: Some(hex::encode(keypair.verifying_key().as_bytes())),
        encryption_secret_key: None,
        encryption_public_key: Some(hex::encode(keypair.encryption_public_key().as_bytes())),
        kdf: Some("argon2id-chacha20poly1305".into()),
        argon2_salt: Some(hex::encode(salt)),
        argon2_m_cost: Some(ARGON2_M_COST),
        argon2_t_cost: Some(ARGON2_T_COST),
        argon2_p_cost: Some(ARGON2_P_COST),
        nonce: Some(hex::encode(nonce)),
        ciphertext: Some(hex::encode(ct)),
    })
}

pub fn write_keystore_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<(), CryptoError> {
    let (parent, tmp_path) = staging_path(path)?;

    let write_result = (|| -> Result<(), CryptoError> {
        write_new_private_file(&tmp_path, bytes)?;

        // On Windows, `rename` fails if the destination already exists.
        // Remove the destination first for Windows; on Unix the rename is atomic.
        #[cfg(windows)]
        if path.exists() {
            std::fs::remove_file(path)?;
        }

        std::fs::rename(&tmp_path, path)?;
        sync_parent_dir(parent)
    })();

    if write_result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }

    write_result
}

/// Write `bytes` to `path` only if nothing exists there yet.
///
/// The bytes are staged in a private temporary file and hard-linked into
/// place, so readers never see a partial keystore and a writer that got
/// there first is never replaced. Filesystems without hard links (such as
/// FAT) refuse the link; there the final file is created exclusively
/// instead, which still never replaces a file but lets a concurrent reader
/// see it before it is fully written.
fn publish_keystore_bytes_without_replacing(path: &Path, bytes: &[u8]) -> Result<(), CryptoError> {
    let (parent, tmp_path) = staging_path(path)?;

    write_new_private_file(&tmp_path, bytes)?;
    let linked = std::fs::hard_link(&tmp_path, path);
    let _ = std::fs::remove_file(&tmp_path);
    match linked {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(keystore_already_exists(path));
        }
        Err(error) if hard_links_unsupported(&error) => write_new_private_file(path, bytes)
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    keystore_already_exists(path)
                } else {
                    error.into()
                }
            })?,
        Err(error) => return Err(error.into()),
    }
    sync_parent_dir(parent)
}

/// Whether a failed `hard_link` means the filesystem has no hard links:
/// `EPERM` on Linux FAT, `EOPNOTSUPP`/`ENOSYS` elsewhere.
fn hard_links_unsupported(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::Unsupported | std::io::ErrorKind::PermissionDenied
    )
}

/// Pick a unique temporary path next to `path`, returning it with the
/// parent directory.
fn staging_path(path: &Path) -> Result<(&Path, PathBuf), CryptoError> {
    let parent = path.parent().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("keystore path {} has no parent directory", path.display()),
        ))
    })?;
    let file_name = path.file_name().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("keystore path {} has no file name", path.display()),
        ))
    })?;
    let tmp_path = parent.join(format!(
        ".{}.tmp-{}-{}",
        file_name.to_string_lossy(),
        std::process::id(),
        rand::random::<u64>()
    ));
    Ok((parent, tmp_path))
}

/// Create `path` exclusively with mode 0600 on Unix and write `bytes` to it
/// durably. A file this call created is removed again if the write fails;
/// an existing file is never touched.
fn write_new_private_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.create_new(true).write(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.mode(0o600);
    }

    let mut file = options.open(path)?;
    let written = file
        .write_all(bytes)
        .and_then(|()| file.flush())
        .and_then(|()| file.sync_all());
    drop(file);
    if written.is_err() {
        let _ = std::fs::remove_file(path);
    }
    written
}

/// fsync the parent directory on Unix so a rename or link survives a crash.
fn sync_parent_dir(parent: &Path) -> Result<(), CryptoError> {
    #[cfg(unix)]
    {
        // A bare file name has an empty parent: the current directory.
        let dir = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        std::fs::File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = parent;
    Ok(())
}

fn plaintext_keystore(ks: &KeystoreV1) -> Result<OwnerKeypair, CryptoError> {
    let sign_hex = ks
        .signing_secret_key
        .as_deref()
        .ok_or(CryptoError::InvalidKeyMaterial {
            reason: "missing signing_secret_key in plaintext keystore".into(),
        })?;
    let enc_hex = ks
        .encryption_secret_key
        .as_deref()
        .ok_or(CryptoError::InvalidKeyMaterial {
            reason: "missing encryption_secret_key in plaintext keystore".into(),
        })?;

    let sign_bytes = hex::decode(sign_hex).map_err(|e| CryptoError::InvalidKeyMaterial {
        reason: format!("bad signing key hex: {e}"),
    })?;
    let enc_bytes = hex::decode(enc_hex).map_err(|e| CryptoError::InvalidKeyMaterial {
        reason: format!("bad encryption key hex: {e}"),
    })?;

    OwnerKeypair::from_bytes(&sign_bytes, &enc_bytes)
}

fn decrypt_keystore(
    ks: &KeystoreV1,
    passphrase: Option<&str>,
) -> Result<OwnerKeypair, CryptoError> {
    let pass = passphrase.ok_or(CryptoError::DecryptionFailed)?;

    let salt = hex::decode(
        ks.argon2_salt
            .as_deref()
            .ok_or(CryptoError::DecryptionFailed)?,
    )
    .map_err(|_| CryptoError::DecryptionFailed)?;
    let nonce_bytes = hex::decode(ks.nonce.as_deref().ok_or(CryptoError::DecryptionFailed)?)
        .map_err(|_| CryptoError::DecryptionFailed)?;
    let ct = hex::decode(
        ks.ciphertext
            .as_deref()
            .ok_or(CryptoError::DecryptionFailed)?,
    )
    .map_err(|_| CryptoError::DecryptionFailed)?;

    if salt.len() != 16 || nonce_bytes.len() != 12 || ct.is_empty() {
        return Err(CryptoError::DecryptionFailed);
    }

    // Use stored KDF parameters for forward compatibility; fall back to the
    // constants that match argon2 0.5.x defaults for keystores that pre-date
    // explicit parameter storage.
    let m_cost = ks.argon2_m_cost.unwrap_or(ARGON2_M_COST);
    let t_cost = ks.argon2_t_cost.unwrap_or(ARGON2_T_COST);
    let p_cost = ks.argon2_p_cost.unwrap_or(ARGON2_P_COST);
    check_argon2_cost_ceiling(m_cost, t_cost, p_cost)?;

    let sym_key = derive_key(pass, &salt, m_cost, t_cost, p_cost)?;
    let key = chacha20poly1305::Key::from(*sym_key);
    let cipher = ChaCha20Poly1305::new(&key);
    let nonce_bytes: [u8; 12] = nonce_bytes
        .try_into()
        .map_err(|_| CryptoError::DecryptionFailed)?;
    let nonce = chacha20poly1305::Nonce::from(nonce_bytes);
    let plaintext = cipher
        .decrypt(&nonce, ct.as_ref())
        .map_err(|_| CryptoError::DecryptionFailed)?;

    let payload: SecretPayload =
        serde_json::from_slice(&plaintext).map_err(|_| CryptoError::DecryptionFailed)?;

    let sign_bytes =
        hex::decode(&payload.signing_secret_key).map_err(|_| CryptoError::DecryptionFailed)?;
    let enc_bytes =
        hex::decode(&payload.encryption_secret_key).map_err(|_| CryptoError::DecryptionFailed)?;

    OwnerKeypair::from_bytes(&sign_bytes, &enc_bytes)
}

/// Validate that stored metadata fields match the loaded keypair.
///
/// Detects tampered keystores where the stored `owner_id`, `signing_public_key`,
/// or `encryption_public_key` don't match what's derived from the secret keys.
fn validate_keystore_metadata(ks: &KeystoreV1, keypair: &OwnerKeypair) -> Result<(), CryptoError> {
    let derived_owner_id = keypair.owner_id();
    if ks.owner_id != derived_owner_id {
        return Err(CryptoError::VerificationFailed {
            reason: "owner_id does not match derived signing key".into(),
        });
    }

    if let Some(ref stored_spk) = ks.signing_public_key {
        let expected = hex::encode(keypair.verifying_key().as_bytes());
        if *stored_spk != expected {
            return Err(CryptoError::VerificationFailed {
                reason: "signing_public_key does not match derived signing key".into(),
            });
        }
    }

    if let Some(ref stored_epk) = ks.encryption_public_key {
        let expected = hex::encode(keypair.encryption_public_key().as_bytes());
        if *stored_epk != expected {
            return Err(CryptoError::VerificationFailed {
                reason: "encryption_public_key does not match derived encryption key".into(),
            });
        }
    }

    Ok(())
}

/// Refuse Argon2 costs from a keystore file that exceed the ceilings, before
/// any memory is allocated for the KDF.
fn check_argon2_cost_ceiling(m_cost: u32, t_cost: u32, p_cost: u32) -> Result<(), CryptoError> {
    for (name, value, max) in [
        ("argon2_m_cost", m_cost, MAX_ARGON2_M_COST),
        ("argon2_t_cost", t_cost, MAX_ARGON2_T_COST),
        ("argon2_p_cost", p_cost, MAX_ARGON2_P_COST),
    ] {
        if value > max {
            return Err(CryptoError::InvalidKeyMaterial {
                reason: format!("keystore {name} {value} exceeds the limit of {max}"),
            });
        }
    }
    Ok(())
}

fn derive_key(
    passphrase: &str,
    salt: &[u8],
    m_cost: u32,
    t_cost: u32,
    p_cost: u32,
) -> Result<Zeroizing<[u8; 32]>, CryptoError> {
    let params =
        Params::new(m_cost, t_cost, p_cost, Some(32 /* output key bytes */)).map_err(|e| {
            CryptoError::InvalidKeyMaterial {
                reason: format!("invalid argon2 params: {e}"),
            }
        })?;
    let argon2 = Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
    let mut key = Zeroizing::new([0u8; 32]);
    argon2
        .hash_password_into(passphrase.as_bytes(), salt, key.as_mut())
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("argon2 KDF failed: {e}"),
        })?;
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_keystore_path() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mesh-llm-test-{}", rand::random::<u64>()));
        fs::create_dir_all(&dir).unwrap();
        dir.join("owner-keystore.json")
    }

    #[test]
    fn round_trip_plaintext() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();
        let original_id = kp.owner_id();

        save_keystore(&path, &kp, None, false).unwrap();
        let loaded = load_keystore(&path, None).unwrap();

        assert_eq!(original_id, loaded.owner_id());
        assert_eq!(kp.signing_bytes(), loaded.signing_bytes());
        assert_eq!(kp.encryption_bytes(), loaded.encryption_bytes());

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn round_trip_encrypted() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();
        let original_id = kp.owner_id();

        save_keystore(&path, &kp, Some("test-passphrase"), false).unwrap();
        let loaded = load_keystore(&path, Some("test-passphrase")).unwrap();

        assert_eq!(original_id, loaded.owner_id());
        assert_eq!(kp.signing_bytes(), loaded.signing_bytes());
        assert_eq!(kp.encryption_bytes(), loaded.encryption_bytes());

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn encrypted_keystore_without_explicit_kdf_parameters_remains_compatible() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();

        save_keystore(&path, &kp, Some("legacy-passphrase"), false).unwrap();
        let mut raw: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let raw = raw.as_object_mut().unwrap();
        raw.remove("argon2_m_cost");
        raw.remove("argon2_t_cost");
        raw.remove("argon2_p_cost");
        fs::write(&path, serde_json::to_string_pretty(&raw).unwrap()).unwrap();

        let loaded = load_keystore(&path, Some("legacy-passphrase")).unwrap();
        assert_eq!(kp.owner_id(), loaded.owner_id());
        assert_eq!(kp.signing_bytes(), loaded.signing_bytes());
        assert_eq!(kp.encryption_bytes(), loaded.encryption_bytes());

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn wrong_passphrase_fails() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();

        save_keystore(&path, &kp, Some("correct"), false).unwrap();

        let result = load_keystore(&path, Some("wrong"));
        assert!(
            matches!(result, Err(CryptoError::DecryptionFailed)),
            "expected DecryptionFailed, got {result:?}"
        );

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn missing_passphrase_for_encrypted_fails() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();

        save_keystore(&path, &kp, Some("secret"), false).unwrap();

        let result = load_keystore(&path, None);
        assert!(
            matches!(result, Err(CryptoError::DecryptionFailed)),
            "expected DecryptionFailed, got {result:?}"
        );

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn keystore_not_found() {
        let path = PathBuf::from("/tmp/nonexistent-keystore.json");
        let result = load_keystore(&path, None);
        assert!(matches!(result, Err(CryptoError::KeystoreNotFound { .. })));
    }

    #[test]
    fn save_refuses_to_overwrite_by_default() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();

        save_keystore(&path, &kp, None, false).unwrap();

        let kp2 = OwnerKeypair::generate();
        let result = save_keystore(&path, &kp2, None, false);
        assert!(
            matches!(result, Err(CryptoError::KeystoreAlreadyExists { .. })),
            "save without overwrite should fail if file exists"
        );

        // With overwrite=true it should succeed.
        save_keystore(&path, &kp2, None, true).unwrap();
        let loaded = load_keystore(&path, None).unwrap();
        assert_eq!(loaded.owner_id(), kp2.owner_id());

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn load_detects_tampered_owner_id() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();
        save_keystore(&path, &kp, None, false).unwrap();

        // Tamper the owner_id in the on-disk JSON.
        let mut raw: KeystoreV1 =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        raw.owner_id = "0000000000000000000000000000000000000000000000000000000000000000".into();
        fs::write(&path, serde_json::to_string_pretty(&raw).unwrap()).unwrap();

        let result = load_keystore(&path, None);
        assert!(
            matches!(result, Err(CryptoError::VerificationFailed { .. })),
            "tampered owner_id should be detected on load"
        );

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn metadata_without_decryption() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();
        let expected_id = kp.owner_id();

        save_keystore(&path, &kp, Some("secret"), false).unwrap();

        let info = keystore_metadata(&path).unwrap();
        assert_eq!(info.owner_id, expected_id);
        assert!(info.encrypted);

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn metadata_rejects_owner_id_spoofing() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();

        save_keystore(&path, &kp, Some("secret"), false).unwrap();

        let mut raw: KeystoreV1 =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        raw.owner_id = "0000000000000000000000000000000000000000000000000000000000000000".into();
        fs::write(&path, serde_json::to_string_pretty(&raw).unwrap()).unwrap();

        let result = keystore_metadata(&path);
        assert!(
            matches!(result, Err(CryptoError::VerificationFailed { .. })),
            "spoofed owner_id should be rejected"
        );

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    fn set_raw_field(path: &Path, field: &str, value: serde_json::Value) {
        let mut raw: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        raw.as_object_mut().unwrap().insert(field.into(), value);
        fs::write(path, serde_json::to_string_pretty(&raw).unwrap()).unwrap();
    }

    #[test]
    fn encrypted_keystore_rejects_argon2_costs_above_ceiling() {
        let path = temp_keystore_path();
        save_keystore(&path, &OwnerKeypair::generate(), Some("secret"), false).unwrap();
        let original = fs::read_to_string(&path).unwrap();

        for (field, value) in [
            ("argon2_m_cost", MAX_ARGON2_M_COST + 1),
            ("argon2_t_cost", MAX_ARGON2_T_COST + 1),
            ("argon2_p_cost", MAX_ARGON2_P_COST + 1),
        ] {
            fs::write(&path, &original).unwrap();
            set_raw_field(&path, field, value.into());

            let result = load_keystore(&path, Some("secret"));
            assert!(
                matches!(
                    &result,
                    Err(CryptoError::InvalidKeyMaterial { reason }) if reason.contains(field)
                ),
                "{field}={value} must be refused before the KDF runs, got {result:?}"
            );
        }

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn concurrent_saves_without_overwrite_publish_exactly_one_keystore() {
        let path = temp_keystore_path();
        let writers = 8;
        let barrier = std::sync::Barrier::new(writers);

        let outcomes: Vec<(String, Result<(), CryptoError>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..writers)
                .map(|_| {
                    scope.spawn(|| {
                        let kp = OwnerKeypair::generate();
                        barrier.wait();
                        (kp.owner_id(), save_keystore(&path, &kp, None, false))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let winners: Vec<&String> = outcomes
            .iter()
            .filter(|(_, result)| result.is_ok())
            .map(|(owner_id, _)| owner_id)
            .collect();
        assert_eq!(winners.len(), 1, "exactly one writer may publish");
        for (_, result) in outcomes.iter().filter(|(_, result)| result.is_err()) {
            assert!(
                matches!(result, Err(CryptoError::KeystoreAlreadyExists { .. })),
                "losing writers must report the existing keystore, got {result:?}"
            );
        }
        assert_eq!(&load_keystore(&path, None).unwrap().owner_id(), winners[0]);

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn publish_without_replacing_keeps_existing_file_and_cleans_up() {
        let path = temp_keystore_path();
        fs::write(&path, "existing").unwrap();

        let result = publish_keystore_bytes_without_replacing(&path, b"new");

        assert!(matches!(
            result,
            Err(CryptoError::KeystoreAlreadyExists { .. })
        ));
        assert_eq!(fs::read_to_string(&path).unwrap(), "existing");
        let entries = fs::read_dir(path.parent().unwrap()).unwrap().count();
        assert_eq!(entries, 1, "the staging file must be removed");

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[cfg(unix)]
    #[test]
    fn load_resets_group_and_other_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();
        save_keystore(&path, &kp, None, false).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );

        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        keystore_metadata(&path).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600,
            "metadata reads must tighten a readable keystore"
        );

        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        assert_eq!(
            load_keystore(&path, None).unwrap().owner_id(),
            kp.owner_id()
        );
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600,
            "loads must tighten a readable keystore"
        );

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_keystore_loads_and_its_target_is_tightened() {
        use std::os::unix::fs::PermissionsExt;

        // Secrets managers commonly provision the keystore as a symlink.
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();
        save_keystore(&path, &kp, None, false).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        let link = path.with_file_name("linked-keystore.json");
        std::os::unix::fs::symlink(&path, &link).unwrap();

        assert_eq!(
            load_keystore(&link, None).unwrap().owner_id(),
            kp.owner_id()
        );
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn load_rejects_keystore_path_that_is_not_a_file() {
        let path = temp_keystore_path();
        fs::create_dir(&path).unwrap();

        assert!(matches!(
            load_keystore(&path, None),
            Err(CryptoError::InvalidKeyMaterial { .. })
        ));
        assert!(matches!(
            keystore_metadata(&path),
            Err(CryptoError::InvalidKeyMaterial { .. })
        ));

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[cfg(unix)]
    #[test]
    fn save_creates_missing_parent_directories_private() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!("mesh-llm-test-{}", rand::random::<u64>()));
        let nested = root.join("nested");
        let path = nested.join("owner-keystore.json");

        save_keystore(&path, &OwnerKeypair::generate(), None, false).unwrap();

        for dir in [&root, &nested] {
            assert_eq!(
                fs::metadata(dir).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn encrypted_keystore_requires_all_encrypted_fields() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();

        save_keystore(&path, &kp, Some("secret"), false).unwrap();

        let mut raw: KeystoreV1 =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        raw.argon2_salt = None;
        fs::write(&path, serde_json::to_string_pretty(&raw).unwrap()).unwrap();

        let result = load_keystore(&path, Some("secret"));
        assert!(
            matches!(result, Err(CryptoError::DecryptionFailed)),
            "missing encrypted fields should fail cleanly"
        );

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
