//! Admission control for inbound tunnel streams.
//!
//! The tunnel manager spawns a handler task for every `STREAM_TUNNEL_HTTP`
//! bi-stream, and each handler holds the QUIC streams and an inflight
//! request slot until the request completes or its header deadline fires.
//! The QUIC transport allows 1024 concurrent bi-directional streams per
//! connection, so an authenticated peer could otherwise keep an unbounded
//! number of handler tasks alive at once (#2206). The HTTP ingress listeners
//! bound their accepted connections with
//! [`crate::network::openai::accept::ConnectionSlots`]; this module does the
//! same for the tunneled ingress: a global quota and a per-peer quota, both
//! claimed before any per-stream work is spawned.
//!
//! Stage-transport streams use the same quotas while they open: from the
//! stream's arrival until its `StageTransportOpen` frame is checked against
//! the stage topology and the local stage accepts the relay connection. A
//! stream over its quota is refused, not queued. Once open, a relay is no
//! longer counted: only a peer that holds a transport edge into the local
//! stage gets that far, and its relays follow the stage's lanes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use iroh::EndpointId;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Most tunneled request handlers a node runs at once, across all peers.
pub(crate) const MAX_TUNNELED_HTTP_INFLIGHT: usize = 256;
/// Most tunneled request handlers one authenticated peer may keep alive at
/// once. Inference is slow and the bottleneck, so a busy routing peer needs
/// only a handful of concurrent requests.
pub(crate) const MAX_TUNNELED_HTTP_INFLIGHT_PER_PEER: usize = 32;
/// Most stage-transport streams a node opens at once, across all peers.
pub(crate) const MAX_STAGE_TRANSPORT_OPENING: usize = 256;
/// Most stage-transport streams one peer may have opening at once. A stage
/// connects one stream per lane, and its lane pool connects the lanes one
/// at a time. Opening usually takes a round trip, but it can wait up to the
/// open-frame deadline plus the local stage-status query's deadline.
pub(crate) const MAX_STAGE_TRANSPORT_OPENING_PER_PEER: usize = 64;

/// Whether one inbound tunnel stream may start its handler.
pub(crate) enum StreamClaim {
    /// The handler holds the permit while the quota covers it.
    Granted(StreamPermit),
    /// A quota is exhausted; the stream should be refused without spawning
    /// work.
    Full,
}

/// A global and per-peer permit for one inbound tunnel stream handler.
///
/// Dropping the permit frees the global slot and the peer's count.
pub(crate) struct StreamPermit {
    _global: OwnedSemaphorePermit,
    peers: Arc<Mutex<HashMap<EndpointId, u32>>>,
    peer: EndpointId,
}

impl Drop for StreamPermit {
    fn drop(&mut self) {
        let mut peers = self.peers.lock().unwrap();
        if let Some(count) = peers.get_mut(&self.peer) {
            *count -= 1;
            if *count == 0 {
                peers.remove(&self.peer);
            }
        }
    }
}

/// The global and per-peer quotas of one kind of inbound tunnel stream.
#[derive(Clone)]
pub(crate) struct StreamAdmission {
    global: Arc<Semaphore>,
    peers: Arc<Mutex<HashMap<EndpointId, u32>>>,
    per_peer: u32,
}

impl StreamAdmission {
    /// Quotas for tunneled HTTP request handlers.
    pub(crate) fn tunneled_http() -> Self {
        Self::with_quotas(
            MAX_TUNNELED_HTTP_INFLIGHT,
            MAX_TUNNELED_HTTP_INFLIGHT_PER_PEER,
        )
    }

    /// Quotas for stage-transport streams that have not opened yet.
    pub(crate) fn stage_transport_opening() -> Self {
        Self::with_quotas(
            MAX_STAGE_TRANSPORT_OPENING,
            MAX_STAGE_TRANSPORT_OPENING_PER_PEER,
        )
    }

    fn with_quotas(global: usize, per_peer: usize) -> Self {
        Self {
            global: Arc::new(Semaphore::new(global)),
            peers: Arc::new(Mutex::new(HashMap::new())),
            per_peer: u32::try_from(per_peer).unwrap_or(u32::MAX),
        }
    }

    /// Claims a global and a per-peer permit for one inbound stream, before
    /// the handler task is spawned.
    pub(crate) fn claim(&self, peer: EndpointId) -> StreamClaim {
        let permit = match Arc::clone(&self.global).try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => return StreamClaim::Full,
        };
        let mut peers = self.peers.lock().unwrap();
        let over_peer_quota = peers
            .get(&peer)
            .is_some_and(|count| *count >= self.per_peer);
        if over_peer_quota {
            drop(permit);
            return StreamClaim::Full;
        }
        *peers.entry(peer).or_insert(0) += 1;
        StreamClaim::Granted(StreamPermit {
            _global: permit,
            peers: Arc::clone(&self.peers),
            peer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(index: u8) -> EndpointId {
        EndpointId::from(iroh::SecretKey::from_bytes(&[index; 32]).public())
    }

    #[test]
    fn a_peer_is_refused_once_it_reaches_the_per_peer_quota() {
        let admission = StreamAdmission::tunneled_http();
        let mut held = Vec::new();
        for _ in 0..MAX_TUNNELED_HTTP_INFLIGHT_PER_PEER {
            match admission.claim(peer(1)) {
                StreamClaim::Granted(permit) => held.push(permit),
                StreamClaim::Full => panic!("under quota the claim must be granted"),
            }
        }
        assert!(
            matches!(admission.claim(peer(1)), StreamClaim::Full),
            "the peer's (MAX_TUNNELED_HTTP_INFLIGHT_PER_PEER + 1)-th request is refused"
        );
        // Another peer is unaffected while its own quota is free.
        assert!(matches!(admission.claim(peer(2)), StreamClaim::Granted(_)));
        drop(held.pop().unwrap());
        assert!(matches!(admission.claim(peer(1)), StreamClaim::Granted(_)));
    }

    #[test]
    fn the_global_quota_is_refused_across_many_peers() {
        let admission = StreamAdmission::tunneled_http();
        let mut held = Vec::new();
        // One peer per claim: the per-peer quota never trips, so only the
        // global quota can refuse.
        for index in 0..MAX_TUNNELED_HTTP_INFLIGHT {
            match admission.claim(peer((index % 251) as u8)) {
                StreamClaim::Granted(permit) => held.push(permit),
                StreamClaim::Full => {
                    panic!("claim {index} must be granted under the global quota")
                }
            }
        }
        assert!(
            matches!(admission.claim(peer(0)), StreamClaim::Full),
            "the global quota refuses beyond MAX_TUNNELED_HTTP_INFLIGHT"
        );
    }

    #[test]
    fn stage_transport_opening_has_its_own_per_peer_quota() {
        let admission = StreamAdmission::stage_transport_opening();
        let mut held = Vec::new();
        for _ in 0..MAX_STAGE_TRANSPORT_OPENING_PER_PEER {
            match admission.claim(peer(1)) {
                StreamClaim::Granted(permit) => held.push(permit),
                StreamClaim::Full => panic!("under quota the claim must be granted"),
            }
        }
        assert!(matches!(admission.claim(peer(1)), StreamClaim::Full));
        assert!(matches!(admission.claim(peer(2)), StreamClaim::Granted(_)));
        // The HTTP ingress quotas are separate.
        assert!(matches!(
            StreamAdmission::tunneled_http().claim(peer(1)),
            StreamClaim::Granted(_)
        ));
        drop(held.pop().unwrap());
        assert!(matches!(admission.claim(peer(1)), StreamClaim::Granted(_)));
    }
}
