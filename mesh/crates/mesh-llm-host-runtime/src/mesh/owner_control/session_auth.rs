//! Authorization of one owner-control stream for as long as it stays open.
//!
//! The handshake names the requester's ownership certificate once, but a
//! stream can carry many requests, and a config watch can stay open for days.
//! The certificate can expire, and the trust store can revoke its owner, cert
//! or node ID, in that time, so every request, and an open watch on an
//! interval, is checked again against the current local owner, trust store and
//! clock.

use super::{Node, owner_control_error_envelope};
use crate::crypto::{ControlPlaneAuthError, verify_control_plane_peer_ownership};
use crate::mesh::current_time_unix_ms;
use iroh::EndpointId;
use std::time::Duration;

/// How often an open config watch checks its requester again.
pub(crate) const OWNER_CONTROL_WATCH_REAUTH_INTERVAL: Duration = Duration::from_secs(5);

/// The requester of one owner-control stream, as its handshake identified it.
pub(crate) struct OwnerControlSession {
    pub(crate) remote: EndpointId,
    ownership: Option<crate::proto::node::SignedNodeOwnership>,
}

impl OwnerControlSession {
    pub(crate) fn new(
        remote: EndpointId,
        handshake: crate::proto::node::OwnerControlHandshake,
    ) -> Self {
        Self {
            remote,
            ownership: handshake.ownership,
        }
    }
}

impl Node {
    /// Whether `session`'s requester may use owner control now.
    pub(crate) async fn authorize_owner_control_session(
        &self,
        session: &OwnerControlSession,
    ) -> Result<(), ControlPlaneAuthError> {
        let local_owner = self.owner_summary.lock().await.clone();
        let trust_store = self.trust_store.lock().await.clone();
        verify_control_plane_peer_ownership(
            &local_owner,
            session.ownership.as_ref(),
            session.remote.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        )
        .map(|_| ())
    }

    /// The terminal error for a failed authorization, tied to `request_id`
    /// when a request was refused rather than the handshake.
    pub(crate) fn owner_control_auth_error_envelope(
        &self,
        err: &ControlPlaneAuthError,
        request_id: Option<u64>,
    ) -> crate::proto::node::OwnerControlEnvelope {
        let code = match err {
            ControlPlaneAuthError::MissingRemoteOwnerAttestation
            | ControlPlaneAuthError::RemoteOwnershipInvalid { .. } => {
                crate::proto::node::OwnerControlErrorCode::InvalidHandshake
            }
            ControlPlaneAuthError::TargetNodeMismatch { .. } => {
                crate::proto::node::OwnerControlErrorCode::TargetNodeMismatch
            }
            ControlPlaneAuthError::MissingLocalOwnerIdentity { .. }
            | ControlPlaneAuthError::RemoteOwnerMismatch { .. }
            | ControlPlaneAuthError::UnsupportedTrustPolicy { .. } => {
                crate::proto::node::OwnerControlErrorCode::Unauthorized
            }
        };
        owner_control_error_envelope(code, request_id, None, err.to_string())
    }

    /// Checks `session` again and, when it no longer passes, ends the stream
    /// with the authorization error. Returns whether the stream may go on.
    pub(crate) async fn reauthorize_owner_control_session(
        &self,
        session: &OwnerControlSession,
        send: &mut iroh::endpoint::SendStream,
        request_id: u64,
    ) -> bool {
        let Err(error) = self.authorize_owner_control_session(session).await else {
            return true;
        };
        tracing::info!(
            "ending owner-control stream from {}: {error}",
            session.remote.fmt_short()
        );
        let _ = self
            .send_owner_control_terminal_envelope(
                send,
                self.owner_control_auth_error_envelope(&error, Some(request_id)),
            )
            .await;
        false
    }
}
