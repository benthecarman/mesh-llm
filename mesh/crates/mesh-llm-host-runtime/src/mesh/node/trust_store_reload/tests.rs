use super::*;
use crate::crypto::{
    DEFAULT_NODE_CERT_LIFETIME_SECS, OwnerKeypair, OwnershipStatus, TrustPolicy, save_trust_store,
    sign_node_ownership,
};
use crate::mesh::NodeRole;
use mesh_llm_identity::OwnerId;

fn test_owner(seed: u8) -> OwnerKeypair {
    OwnerKeypair::from_bytes(&[seed; 32], &[seed.wrapping_add(1); 32]).unwrap()
}

/// A node enforcing `policy`, with one admitted peer whose certificate
/// `owner` signed.
async fn node_with_owned_peer(policy: TrustPolicy, owner: &OwnerKeypair) -> (Node, EndpointId) {
    let mut node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    node.trust_policy = policy;
    let peer_id = iroh::SecretKey::generate().public();
    let attestation = sign_node_ownership(
        owner,
        peer_id.as_bytes(),
        current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        None,
        None,
    )
    .unwrap();
    let mut peer = crate::mesh::tests::make_test_peer(peer_id, None, 0);
    peer.owner_summary = verify_node_ownership(
        Some(&attestation),
        peer_id.as_bytes(),
        &TrustStore::default(),
        policy,
        current_time_unix_ms(),
    );
    assert_eq!(peer.owner_summary.status, OwnershipStatus::Verified);
    peer.owner_attestation = Some(attestation);
    node.state.lock().await.peers.insert(peer_id, peer);
    (node, peer_id)
}

fn revoking(owner: &OwnerKeypair) -> TrustStore {
    let mut store = TrustStore::default();
    store.revoke_owner(&OwnerId::parse(&owner.owner_id()).unwrap(), None);
    store
}

#[tokio::test]
async fn revoking_an_admitted_peers_owner_removes_it() {
    let owner = test_owner(0x31);
    let (node, peer_id) = node_with_owned_peer(TrustPolicy::RequireOwned, &owner).await;

    assert!(node.apply_trust_store(revoking(&owner)).await);

    assert!(!node.state.lock().await.peers.contains_key(&peer_id));
    assert_eq!(
        node.state.lock().await.policy_rejected_peers.get(&peer_id),
        Some(&OwnershipStatus::RevokedOwner)
    );
}

#[tokio::test]
async fn an_open_policy_keeps_the_peer_but_reports_the_revocation() {
    let owner = test_owner(0x33);
    let (node, peer_id) = node_with_owned_peer(TrustPolicy::Off, &owner).await;

    assert!(node.apply_trust_store(revoking(&owner)).await);

    let state = node.state.lock().await;
    let peer = state.peers.get(&peer_id).expect("Off admits every peer");
    assert_eq!(peer.owner_summary.status, OwnershipStatus::RevokedOwner);
}

#[tokio::test]
async fn an_unchanged_store_is_not_reapplied() {
    let owner = test_owner(0x35);
    let (node, peer_id) = node_with_owned_peer(TrustPolicy::RequireOwned, &owner).await;

    assert!(!node.apply_trust_store(TrustStore::default()).await);
    assert!(node.state.lock().await.peers.contains_key(&peer_id));
}

#[tokio::test]
async fn a_revoked_local_owner_stops_verifying_this_node() {
    let owner = test_owner(0x37);
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let attestation = sign_node_ownership(
        &owner,
        node.id().as_bytes(),
        current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        None,
        None,
    )
    .unwrap();
    *node.owner_attestation.lock().await = Some(attestation);

    node.apply_trust_store(revoking(&owner)).await;

    let summary = node.owner_summary.lock().await.clone();
    assert_eq!(summary.status, OwnershipStatus::RevokedOwner);
    assert!(!summary.verified);
}

/// The CLI writes the file from another process; the node picks the edit up
/// on its next reload and keeps the last good store when the file is bad.
#[tokio::test]
async fn reload_reads_cli_edits_and_keeps_the_last_good_store() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trusted-owners.json");
    let owner = test_owner(0x39);
    let (node, peer_id) = node_with_owned_peer(TrustPolicy::RequireOwned, &owner).await;
    let mut reload = TrustStoreReload::new(TrustStoreSource::optional_file(path.clone(), vec![]));

    reload.reload_into(&node).await;
    assert!(node.state.lock().await.peers.contains_key(&peer_id));

    save_trust_store(&path, &revoking(&owner)).unwrap();
    reload.reload_into(&node).await;
    assert!(!node.state.lock().await.peers.contains_key(&peer_id));

    std::fs::write(&path, b"{ not json").unwrap();
    reload.reload_into(&node).await;
    assert!(
        node.trust_store
            .lock()
            .await
            .owner_is_revoked(&owner.owner_id())
    );

    // A file that goes missing must not read as an empty store: that would
    // forget the revocation until the file came back.
    std::fs::remove_file(&path).unwrap();
    reload.reload_into(&node).await;
    assert!(
        node.trust_store
            .lock()
            .await
            .owner_is_revoked(&owner.owner_id())
    );
}

#[test]
fn sources_merge_extra_owners_and_require_explicit_files() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing.json");
    let owner = test_owner(0x3b).owner_id();

    let optional =
        TrustStoreSource::optional_file(missing.clone(), vec![owner.to_ascii_uppercase()])
            .load()
            .unwrap();
    assert!(optional.trusts_owner(&owner));

    assert!(TrustStoreSource::required_file(missing).load().is_err());
}
