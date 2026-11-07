use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};

use super::error::CryptoError;

/// Owner keypair: Ed25519 signing key + X25519 encryption key.
#[derive(Debug)]
pub struct OwnerKeypair {
    pub(crate) signing: SigningKey,
    pub(crate) encryption: crypto_box::SecretKey,
}

impl OwnerKeypair {
    /// Generate a new random owner keypair.
    pub fn generate() -> Self {
        // ed25519-dalek (rand_core 0.9) and crypto_box (rand_core 0.6)
        // need different RNG types due to version mismatch.
        let signing = SigningKey::generate(&mut rand::rng());
        let encryption = crypto_box::SecretKey::generate(&mut crypto_box::aead::OsRng);
        Self {
            signing,
            encryption,
        }
    }

    /// Derive the stable owner ID from the signing public key.
    ///
    /// Returns `sha256(signing_public_key_bytes)` as a 64-char lowercase hex string.
    pub fn owner_id(&self) -> String {
        owner_id_from_verifying_key(&self.signing.verifying_key())
    }

    /// The Ed25519 verifying (public) key for signature verification.
    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing.verifying_key()
    }

    /// The X25519 public key for encrypting messages to this owner.
    pub fn encryption_public_key(&self) -> crypto_box::PublicKey {
        self.encryption.public_key()
    }

    /// Reconstruct from raw key bytes (used by keystore deserialization).
    pub fn from_bytes(signing_bytes: &[u8], encryption_bytes: &[u8]) -> Result<Self, CryptoError> {
        let signing_arr: [u8; 32] =
            signing_bytes
                .try_into()
                .map_err(|_| CryptoError::InvalidKeyMaterial {
                    reason: "signing key must be 32 bytes".into(),
                })?;
        let encryption_arr: [u8; 32] =
            encryption_bytes
                .try_into()
                .map_err(|_| CryptoError::InvalidKeyMaterial {
                    reason: "encryption key must be 32 bytes".into(),
                })?;

        let signing = SigningKey::from_bytes(&signing_arr);
        let encryption = crypto_box::SecretKey::from(encryption_arr);

        Ok(Self {
            signing,
            encryption,
        })
    }

    /// Raw signing secret key bytes (for keystore serialization).
    pub fn signing_bytes(&self) -> &[u8; 32] {
        self.signing.as_bytes()
    }

    /// Raw encryption secret key bytes (for keystore serialization).
    pub fn encryption_bytes(&self) -> [u8; 32] {
        self.encryption.to_bytes()
    }

    /// Sign arbitrary domain-separated bytes with the owner signing key.
    pub fn sign_bytes(&self, bytes: &[u8]) -> [u8; 64] {
        self.signing.sign(bytes).to_bytes()
    }
}

impl Clone for OwnerKeypair {
    fn clone(&self) -> Self {
        Self {
            signing: self.signing.clone(),
            encryption: crypto_box::SecretKey::from(self.encryption.to_bytes()),
        }
    }
}

// Both ed25519_dalek::SigningKey and crypto_box::SecretKey implement Zeroize on drop.

/// Derive owner ID from a verifying key (public operation, no secret needed).
pub fn owner_id_from_verifying_key(vk: &VerifyingKey) -> String {
    let hash = Sha256::digest(vk.as_bytes());
    hex::encode(hash)
}

/// The canonical form of an owner ID: 64 lowercase hex characters, the
/// SHA-256 of the owner's signing key.
///
/// Certificates always carry this form, so an ID typed by a person (in another
/// case, or with stray whitespace) is parsed into it before it is stored or
/// compared. A value that is not 32 bytes of hex is rejected rather than stored
/// as an entry that can never match.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OwnerId(String);

impl OwnerId {
    pub fn parse(value: &str) -> Result<Self, CryptoError> {
        parse_hex_identifier("owner ID", value, 32).map(Self)
    }

    pub fn from_verifying_key(vk: &VerifyingKey) -> Self {
        Self(owner_id_from_verifying_key(vk))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl std::fmt::Display for OwnerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::str::FromStr for OwnerId {
    type Err = CryptoError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

/// Trim and lowercase a hex identifier of exactly `byte_len` bytes.
pub(crate) fn parse_hex_identifier(
    kind: &'static str,
    value: &str,
    byte_len: usize,
) -> Result<String, CryptoError> {
    let normalized = value.trim().to_ascii_lowercase();
    let invalid = |reason: String| CryptoError::InvalidIdentifier { kind, reason };
    if normalized.len() != byte_len * 2 {
        return Err(invalid(format!(
            "expected {} hex characters, got {}",
            byte_len * 2,
            normalized.len()
        )));
    }
    if !normalized.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid("expected only hex characters".into()));
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_generation_produces_valid_owner_id() {
        let kp = OwnerKeypair::generate();
        let id = kp.owner_id();
        assert_eq!(id.len(), 64, "owner_id should be 64 hex chars");
        assert!(
            id.chars().all(|c| c.is_ascii_hexdigit()),
            "owner_id should be hex"
        );
    }

    #[test]
    fn owner_id_is_deterministic_from_public_key() {
        let kp = OwnerKeypair::generate();
        let id1 = kp.owner_id();
        let id2 = owner_id_from_verifying_key(&kp.verifying_key());
        assert_eq!(id1, id2);
    }

    #[test]
    fn different_keypairs_produce_different_owner_ids() {
        let kp1 = OwnerKeypair::generate();
        let kp2 = OwnerKeypair::generate();
        assert_ne!(kp1.owner_id(), kp2.owner_id());
    }

    #[test]
    fn owner_id_parse_normalizes_case_and_whitespace() {
        let kp = OwnerKeypair::generate();
        let typed = format!("  {}\n", kp.owner_id().to_ascii_uppercase());
        let parsed = OwnerId::parse(&typed).unwrap();
        assert_eq!(parsed.as_str(), kp.owner_id());
        assert_eq!(parsed, OwnerId::from_verifying_key(&kp.verifying_key()));
    }

    #[test]
    fn owner_id_parse_rejects_malformed_values() {
        let kp = OwnerKeypair::generate();
        let owner_id = kp.owner_id();
        for bad in [
            String::new(),
            owner_id[..63].to_string(),
            format!("{owner_id}0"),
            format!("{}g", &owner_id[..63]),
            "owner-a".to_string(),
        ] {
            assert!(
                matches!(
                    OwnerId::parse(&bad),
                    Err(CryptoError::InvalidIdentifier {
                        kind: "owner ID",
                        ..
                    })
                ),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn round_trip_from_bytes() {
        let kp = OwnerKeypair::generate();
        let signing = kp.signing_bytes().to_vec();
        let encryption = kp.encryption_bytes().to_vec();

        let restored = OwnerKeypair::from_bytes(&signing, &encryption).unwrap();
        assert_eq!(kp.owner_id(), restored.owner_id());
        assert_eq!(
            kp.encryption_public_key().as_bytes(),
            restored.encryption_public_key().as_bytes()
        );
    }
}
