// A trust-store change reaches owner-control streams that are already open.

fn get_config_request(
    request_id: u64,
    requester: EndpointId,
    target: EndpointId,
) -> crate::proto::node::OwnerControlEnvelope {
    crate::proto::node::OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: Some(crate::proto::node::OwnerControlRequest {
            get_config: Some(crate::proto::node::OwnerControlGetConfigRequest {
                requester_node_id: requester.as_bytes().to_vec(),
                target_node_id: target.as_bytes().to_vec(),
            }),
            ..owner_control_request(request_id)
        }),
        response: None,
        error: None,
    }
}

fn revoking_node_id(id: EndpointId) -> TrustStore {
    let mut store = TrustStore::default();
    store.revoke_node_id(hex::encode(id.as_bytes()), Some("lost".into()));
    store
}

fn assert_refused_as_revoked(envelope: &crate::proto::node::OwnerControlEnvelope, request_id: u64) {
    assert!(envelope.response.is_none(), "a revoked requester gets no response");
    let error = envelope.error.as_ref().expect("revoked requester gets an error");
    assert_eq!(error.request_id, Some(request_id));
    assert_eq!(
        error.code,
        crate::proto::node::OwnerControlErrorCode::InvalidHandshake as i32
    );
    assert!(error.message.contains("RevokedNodeId"), "{}", error.message);
}

#[tokio::test]
async fn owner_control_refuses_later_requests_once_the_requester_is_revoked() -> Result<()> {
    let owner_keypair = test_owner_keypair(0xa1, 0xa2);
    let tmp = tempfile::tempdir()?;
    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, tmp.path()).await?;
    let (_endpoint, mut send, mut recv, requester) =
        open_owner_control_stream(&server, &owner_keypair).await?;

    write_len_prefixed(
        &mut send,
        &get_config_request(1, requester, server.id()).encode_to_vec(),
    )
    .await?;
    let first = read_owner_control_envelope(&mut recv).await?;
    assert!(first.response.is_some(), "the handshake was authorized");

    assert!(server.apply_trust_store(revoking_node_id(requester)).await);
    write_len_prefixed(
        &mut send,
        &get_config_request(2, requester, server.id()).encode_to_vec(),
    )
    .await?;
    let refused = read_owner_control_envelope(&mut recv).await?;
    assert_refused_as_revoked(&refused, 2);

    server.shutdown_control_listener().await;
    Ok(())
}

#[tokio::test]
async fn an_open_config_watch_ends_once_the_requester_is_revoked() -> Result<()> {
    let owner_keypair = test_owner_keypair(0xa3, 0xa4);
    let tmp = tempfile::tempdir()?;
    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, tmp.path()).await?;
    let (_endpoint, mut send, mut recv, requester) =
        open_owner_control_stream(&server, &owner_keypair).await?;
    write_len_prefixed(
        &mut send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(crate::proto::node::OwnerControlRequest {
                watch_config: Some(crate::proto::node::OwnerControlWatchConfigRequest {
                    requester_node_id: requester.as_bytes().to_vec(),
                    target_node_id: server.id().as_bytes().to_vec(),
                    include_snapshot: false,
                }),
                ..owner_control_request(7)
            }),
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;
    let accepted = read_owner_control_envelope(&mut recv).await?;
    assert!(accepted.response.is_some(), "the watch was accepted");

    assert!(server.apply_trust_store(revoking_node_id(requester)).await);
    // The watch checks again within one reauthorization interval; allow
    // generous slack for a loaded machine.
    let ended = tokio::time::timeout(
        crate::mesh::owner_control::OWNER_CONTROL_WATCH_REAUTH_INTERVAL * 6,
        read_owner_control_envelope(&mut recv),
    )
    .await??;
    assert_refused_as_revoked(&ended, 7);

    server.shutdown_control_listener().await;
    Ok(())
}

/// The check every request and open watch runs reads the clock when it runs,
/// so a certificate that was valid at the handshake stops working once it
/// expires.
#[tokio::test]
async fn owner_control_authorization_refuses_an_expired_certificate() -> Result<()> {
    use crate::mesh::owner_control::OwnerControlSession;

    let owner_keypair = test_owner_keypair(0xa5, 0xa6);
    let tmp = tempfile::tempdir()?;
    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, tmp.path()).await?;
    let requester = SecretKey::generate().public();
    let session = |expires_at_unix_ms| -> Result<OwnerControlSession> {
        let ownership = sign_node_ownership(
            &owner_keypair,
            requester.as_bytes(),
            expires_at_unix_ms,
            None,
            None,
        )?;
        Ok(OwnerControlSession::new(
            requester,
            crate::proto::node::OwnerControlHandshake {
                ownership: Some(proto_signed_node_ownership(&ownership)),
            },
        ))
    };
    let now = current_time_unix_ms();

    assert!(
        server
            .authorize_owner_control_session(&session(now + 60_000)?)
            .await
            .is_ok()
    );
    let error = server
        .authorize_owner_control_session(&session(now.saturating_sub(1))?)
        .await
        .unwrap_err();
    assert!(
        matches!(
            error,
            crate::crypto::ControlPlaneAuthError::RemoteOwnershipInvalid {
                status: OwnershipStatus::Expired,
                ..
            }
        ),
        "{error}"
    );

    server.shutdown_control_listener().await;
    Ok(())
}
