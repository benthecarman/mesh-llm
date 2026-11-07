//! Picks up trust-store edits made while the node runs.
//!
//! `mesh-llm auth revoke-owner`, `revoke-node`, `rotate-node --revoke-current`
//! and `trust add/remove` write the trust store file from another process. The
//! node rereads that file on a short interval, and a change takes effect on
//! peers it has already admitted, not only on later gossip. The trust policy
//! stays the one the node started with.

use super::Node;
use crate::crypto::{TrustStore, load_trust_store};
use crate::mesh::{current_time_unix_ms, verify_node_ownership};
use anyhow::{Context, Result};
use iroh::EndpointId;
use std::path::PathBuf;
use std::time::Duration;

/// How often a running node rereads its trust store file.
pub(crate) const TRUST_STORE_RELOAD_INTERVAL: Duration = Duration::from_secs(5);

/// Where a node's trust store comes from, so it can be read again.
#[derive(Debug, Clone)]
pub(crate) struct TrustStoreSource {
    path: PathBuf,
    /// Owners trusted on top of the file, such as `--trust-owner`.
    extra_trusted_owners: Vec<String>,
    /// An explicit membership store must exist; the default store may not.
    require_file: bool,
}

impl TrustStoreSource {
    /// The default store, which reads as empty when the file is missing at
    /// startup.
    pub(crate) fn optional_file(path: PathBuf, extra_trusted_owners: Vec<String>) -> Self {
        Self {
            path,
            extra_trusted_owners,
            require_file: false,
        }
    }

    /// A store named in configuration, which must stay readable.
    pub(crate) fn required_file(path: PathBuf) -> Self {
        Self {
            path,
            extra_trusted_owners: Vec::new(),
            require_file: true,
        }
    }

    pub(crate) fn load(&self) -> Result<TrustStore> {
        if self.require_file && !self.path.exists() {
            anyhow::bail!("trust store {} is missing", self.path.display());
        }
        Ok(load_trust_store(&self.path)
            .with_context(|| format!("Failed to load trust store {}", self.path.display()))?
            .merged_with_trusted_owners(&self.extra_trusted_owners))
    }

    /// The store to install on a reload, or `None` when the optional file is
    /// absent. A missing file then keeps what the node already enforces:
    /// treating it as empty would forget every revocation while the file is
    /// briefly gone, as during a restore of `~/.mesh-llm`.
    fn reload(&self) -> Result<Option<TrustStore>> {
        if !self.require_file && !self.path.exists() {
            return Ok(None);
        }
        self.load().map(Some)
    }
}

/// Rereads one [`TrustStoreSource`] into a node. A file that cannot be read,
/// or that goes missing, keeps the last good store, so a half-written edit or
/// a removed file never drops revocations; the same failure is logged once
/// until it changes.
pub(crate) struct TrustStoreReload {
    source: TrustStoreSource,
    last_error: Option<String>,
    /// Whether the last good store came from the file, so that file going
    /// missing is worth a warning; a node that never had one stays quiet.
    loaded_from_file: bool,
}

impl TrustStoreReload {
    pub(crate) fn new(source: TrustStoreSource) -> Self {
        let loaded_from_file = source.path.exists();
        Self {
            source,
            last_error: None,
            loaded_from_file,
        }
    }

    pub(crate) async fn reload_into(&mut self, node: &Node) {
        match self.source.reload() {
            Ok(Some(store)) => {
                self.last_error = None;
                self.loaded_from_file = true;
                if node.apply_trust_store(store).await {
                    tracing::info!(
                        path = %self.source.path.display(),
                        "reloaded the trust store"
                    );
                }
            }
            Ok(None) if self.loaded_from_file => self.warn_once(format!(
                "trust store {} is missing; it takes effect again once it is restored",
                self.source.path.display()
            )),
            Ok(None) => {}
            Err(error) => self.warn_once(format!("{error:#}")),
        }
    }

    fn warn_once(&mut self, message: String) {
        if self.last_error.as_deref() != Some(message.as_str()) {
            tracing::warn!("keeping the current trust store, retrying: {message}");
        }
        self.last_error = Some(message);
    }
}

impl Node {
    /// Rereads `source` every [`TRUST_STORE_RELOAD_INTERVAL`] for as long as
    /// the returned task runs.
    pub(crate) fn start_trust_store_reload(
        &self,
        source: TrustStoreSource,
    ) -> tokio::task::JoinHandle<()> {
        let node = self.clone();
        let mut reload = TrustStoreReload::new(source);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(TRUST_STORE_RELOAD_INTERVAL).await;
                reload.reload_into(&node).await;
            }
        })
    }

    /// Installs `store` and enforces it on this node, its admitted peers and
    /// its memberships. Returns whether this node's store changed.
    pub(crate) async fn apply_trust_store(&self, store: TrustStore) -> bool {
        if !self.install_trust_store(store).await {
            return false;
        }
        self.push_revocations_to_memberships().await;
        true
    }

    /// Installs `store` on this node alone and enforces it on the local
    /// owner summary and admitted peers. A membership node has no memberships
    /// of its own, so this is all a pushed revocation needs there.
    pub(super) async fn install_trust_store(&self, store: TrustStore) -> bool {
        {
            let mut current = self.trust_store.lock().await;
            if *current == store {
                return false;
            }
            *current = store;
        }
        self.refresh_local_owner_summary().await;
        self.enforce_trust_store_on_admitted_peers().await;
        true
    }

    /// A revoked local owner or certificate stops counting as verified, which
    /// owner control and artifact transfer check.
    async fn refresh_local_owner_summary(&self) {
        let attestation = self.owner_attestation.lock().await.clone();
        let summary = verify_node_ownership(
            attestation.as_ref(),
            self.id().as_bytes(),
            &*self.trust_store.lock().await,
            crate::mesh::TrustPolicy::Off,
            current_time_unix_ms(),
        );
        *self.owner_summary.lock().await = summary;
    }

    /// Re-verifies every admitted peer's certificate against the current store.
    /// A peer the policy now refuses is removed, as gossip would remove it;
    /// the others get their ownership status refreshed.
    async fn enforce_trust_store_on_admitted_peers(&self) {
        let peers: Vec<(EndpointId, Option<crate::crypto::SignedNodeOwnership>)> = {
            let state = self.state.lock().await;
            state
                .peers
                .iter()
                .map(|(id, peer)| (*id, peer.owner_attestation.clone()))
                .collect()
        };
        let store = self.trust_store.lock().await.clone();
        let now = current_time_unix_ms();
        for (id, attestation) in peers {
            let summary = verify_node_ownership(
                attestation.as_ref(),
                id.as_bytes(),
                &store,
                self.trust_policy,
                now,
            );
            if self.reject_direct_peer_for_policy(id, &summary).await {
                continue;
            }
            if let Some(peer) = self.state.lock().await.peers.get_mut(&id) {
                peer.owner_summary = summary;
            }
        }
    }
}

#[cfg(test)]
mod tests;
