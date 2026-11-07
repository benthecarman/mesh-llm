//! Per-connection bounds on the requests a plugin starts against the host.
//!
//! Each plugin request is handled on its own task so the connection reader
//! keeps moving. Without a bound, one plugin could start any number of those
//! tasks. A request over its bound is refused with an error response; the
//! reader never waits for capacity, because a handler may itself be waiting
//! on a reply that only the reader can deliver.
//!
//! Every request is answered or its connection ends. A refusal is queued by
//! a small task of its own, so it is not lost when the outbound queue is
//! full. When even those refusal tasks are all busy, the plugin keeps
//! starting requests while it does not read the replies, and the connection
//! is closed instead of leaving requests that are never answered.
use super::*;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Host control requests (mesh streams, peer blocks, plugin keys, model
/// requirements and model asset jobs) one connection may have in flight.
pub(super) const CONTROL_REQUEST_CAPACITY: usize = 64;

/// Bridge RPCs (inference and MCP calls) one connection may have in flight.
///
/// A plugin such as mixture-of-agents sends one inference RPC for each
/// candidate of each user request, and each holds its slot for the whole
/// inference. The bound is therefore as large as the number of connections
/// the host's own OpenAI ingress serves at once.
pub(super) const BRIDGE_REQUEST_CAPACITY: usize =
    crate::network::openai::accept::MAX_INGRESS_CONNECTIONS;

/// Refusals one pool may have waiting for room in the outbound queue.
pub(super) const REFUSAL_CAPACITY: usize = 64;

/// A plugin-started request that holds one of its connection's slots until
/// its handler task drops it.
pub(super) struct Admitted {
    pub(super) request_id: u64,
    pub(super) permit: OwnedSemaphorePermit,
}

#[cfg(test)]
impl Admitted {
    /// A slot from a fresh, private pool, for driving one handler directly.
    pub(super) fn for_test(request_id: u64) -> Self {
        let permit = Arc::new(Semaphore::new(1))
            .try_acquire_owned()
            .expect("a fresh pool has a slot");
        Self { request_id, permit }
    }
}

/// One pool of slots for plugin-started requests on one connection.
pub(super) struct RpcAdmission {
    capacity: Arc<Semaphore>,
    limit: usize,
    refusals: Arc<Semaphore>,
}

impl RpcAdmission {
    pub(super) fn new(limit: usize, refusal_limit: usize) -> Self {
        Self {
            capacity: Arc::new(Semaphore::new(limit)),
            limit,
            refusals: Arc::new(Semaphore::new(refusal_limit)),
        }
    }

    /// Takes a slot for request `request_id`, or answers it with an error.
    ///
    /// `Ok(None)` means the request was refused and its refusal is queued.
    /// `Err` means the refusal could not be queued either; the caller must
    /// end the connection.
    pub(super) fn admit(
        &self,
        plugin_name: &str,
        request_id: u64,
        outbound_tx: &mpsc::Sender<proto::Envelope>,
    ) -> Result<Option<Admitted>> {
        if let Ok(permit) = self.capacity.clone().try_acquire_owned() {
            return Ok(Some(Admitted { request_id, permit }));
        }
        let Ok(refusal_slot) = self.refusals.clone().try_acquire_owned() else {
            bail!(
                "plugin '{plugin_name}' keeps starting host requests while it does not \
                 read the replies"
            );
        };
        tracing::warn!(
            plugin = %plugin_name,
            request_id,
            limit = self.limit,
            "plugin has too many host requests in flight; refusing request"
        );
        let refusal = refusal_envelope(plugin_name, request_id);
        let outbound_tx = outbound_tx.clone();
        tokio::spawn(async move {
            let _refusal_slot = refusal_slot;
            // Fails only once the connection is gone, and its plugin with it.
            let _ = outbound_tx.send(refusal).await;
        });
        Ok(None)
    }
}

fn refusal_envelope(plugin_name: &str, request_id: u64) -> proto::Envelope {
    proto::Envelope {
        protocol_version: PROTOCOL_VERSION,
        plugin_id: plugin_name.to_string(),
        request_id,
        payload: Some(proto::envelope::Payload::ErrorResponse(
            proto::ErrorResponse {
                code: ErrorCode::INTERNAL_ERROR.0,
                message: format!("plugin '{plugin_name}' has too many host requests in flight"),
                data_json: String::new(),
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WAIT: Duration = Duration::from_secs(30);

    async fn next(outbound_rx: &mut mpsc::Receiver<proto::Envelope>) -> proto::Envelope {
        tokio::time::timeout(WAIT, outbound_rx.recv())
            .await
            .expect("the refusal is sent")
            .expect("the outbound queue stays open")
    }

    #[tokio::test]
    async fn requests_over_the_bound_are_refused_until_a_slot_frees() {
        let admission = RpcAdmission::new(1, 1);
        let (outbound_tx, mut outbound_rx) = mpsc::channel(4);

        let held = admission
            .admit("demo", 1, &outbound_tx)
            .unwrap()
            .expect("the first request fits");
        assert!(admission.admit("demo", 2, &outbound_tx).unwrap().is_none());

        let refusal = next(&mut outbound_rx).await;
        assert_eq!(refusal.request_id, 2);
        assert_eq!(refusal.plugin_id, "demo");
        let Some(proto::envelope::Payload::ErrorResponse(error)) = refusal.payload else {
            panic!("a refused request is answered with an error");
        };
        assert!(error.message.contains("too many host requests"));

        drop(held);
        assert!(admission.admit("demo", 3, &outbound_tx).unwrap().is_some());
        assert!(outbound_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn a_refusal_waits_for_room_in_a_full_outbound_queue() {
        let admission = RpcAdmission::new(0, 1);
        let (outbound_tx, mut outbound_rx) = mpsc::channel(1);
        outbound_tx
            .try_send(proto::Envelope::default())
            .expect("fixture fills the queue");

        assert!(admission.admit("demo", 9, &outbound_tx).unwrap().is_none());
        assert_eq!(next(&mut outbound_rx).await.request_id, 0);
        assert_eq!(next(&mut outbound_rx).await.request_id, 9);
    }

    #[tokio::test]
    async fn a_plugin_that_never_reads_its_refusals_is_disconnected() {
        let admission = RpcAdmission::new(0, 1);
        let (outbound_tx, _outbound_rx) = mpsc::channel(1);
        outbound_tx
            .try_send(proto::Envelope::default())
            .expect("fixture fills the queue");

        assert!(admission.admit("demo", 1, &outbound_tx).unwrap().is_none());
        let error = admission
            .admit("demo", 2, &outbound_tx)
            .err()
            .expect("with every refusal still queued the connection ends");
        assert!(error.to_string().contains("does not read the replies"));
    }
}
