//! One plugin control connection: a read loop and a write loop that share
//! the stream and run side by side until either side ends.
//!
//! Reading and writing must not take turns. A frame read is not cancel-safe
//! (see [`super::frames`]), so it is never raced against outbound sends; and
//! a write that waits for a plugin to drain its socket must not stop the host
//! from reading what that plugin is sending meanwhile.
use super::*;

type ConnectionLoopFn = fn(
    LocalStream,
    mpsc::Receiver<super::proto::Envelope>,
    Arc<Mutex<HashMap<u64, oneshot::Sender<Result<super::proto::Envelope>>>>>,
    mpsc::Sender<PluginMeshEvent>,
    String,
    Arc<Mutex<PluginSummary>>,
    Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    Arc<Mutex<Option<PluginRuntime>>>,
    mpsc::Sender<super::proto::Envelope>,
    u64,
    Arc<AtomicBool>,
) -> Pin<Box<dyn Future<Output = ()> + Send>>;

pub(crate) const CONNECTION_LOOP: ConnectionLoopFn =
    |stream,
     mut outbound_rx,
     pending,
     mesh_tx,
     plugin_name,
     summary,
     rpc_bridge,
     runtime,
     outbound_tx,
     generation,
     peer_state| {
        let authenticated_peer = peer_state.load(Ordering::Acquire);
        Box::pin(async move {
            let mut inbound = ConnectionInbound {
                plugin_name: plugin_name.clone(),
                generation,
                authenticated_peer,
                pending: pending.clone(),
                mesh_tx: mesh_tx.clone(),
                rpc_bridge,
                runtime: runtime.clone(),
                outbound_tx,
                binding_capacity: Arc::new(tokio::sync::Semaphore::new(8)),
                control_admission: rpc_admission::RpcAdmission::new(
                    rpc_admission::CONTROL_REQUEST_CAPACITY,
                    rpc_admission::REFUSAL_CAPACITY,
                ),
                bridge_admission: rpc_admission::RpcAdmission::new(
                    rpc_admission::BRIDGE_REQUEST_CAPACITY,
                    rpc_admission::REFUSAL_CAPACITY,
                ),
                observation: Default::default(),
                status_lane: None,
            };
            // Reads and writes run side by side for the whole connection: a
            // partly read frame is never dropped for an outbound send, and a
            // write that waits on a full plugin socket never stops the host
            // from reading what the plugin sends meanwhile.
            let (mut reader, mut writer) = tokio::io::split(stream);
            let result: Result<()> = tokio::select! {
                result = read_inbound(&mut reader, &mut inbound) => result,
                result = write_outbound(&mut writer, &mut outbound_rx) => result,
            };
            inbound
                .observation
                .end(&crate::plugin::inference_observation::hub());
            drop(inbound);

            let was_active_runtime = {
                let mut runtime = runtime.lock().await;
                if runtime.as_ref().map(|runtime| runtime.generation) == Some(generation) {
                    peer_state.store(false, Ordering::Release);
                    *runtime = None;
                    true
                } else {
                    false
                }
            };

            if was_active_runtime {
                let _ = mesh_tx
                    .send_timeout(
                        PluginMeshEvent::ModelRequirementsWithdraw {
                            plugin_id: plugin_name.clone(),
                        },
                        Duration::from_secs(1),
                    )
                    .await;
                if let Err(err) = result {
                    tracing::warn!(
                        plugin = %plugin_name,
                        error = %err,
                        "Plugin connection closed"
                    );
                }
                // The plugin went away by itself (it crashed or exited), so the
                // supervisor starts it again on its next round. A shutdown or a
                // disable that takes the runtime first never gets here; one that
                // runs once it is gone finds nothing to stop and settles the
                // status itself, which stays: a disabled plugin must stay down.
                let mut summary = summary.lock().await;
                if !matches!(
                    summary.status.as_str(),
                    "shutting down" | "stopped" | "disabled"
                ) {
                    summary.status = "restarting".into();
                    summary.pid = None;
                    summary.error = Some(format!("Plugin '{}' disconnected", plugin_name));
                }
                drop(summary);
                crate::plugin::runtime::status_publication::announce_disconnected(
                    &mesh_tx,
                    &plugin_name,
                    generation,
                );
            }

            let mut pending = pending.lock().await;
            for (_, responder) in pending.drain() {
                let _ = responder.send(Err(anyhow!("Plugin '{}' disconnected", plugin_name)));
            }
        })
    };

pub(crate) use CONNECTION_LOOP as connection_loop;

type PendingResponses = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<super::proto::Envelope>>>>>;

/// What the reader needs to act on the envelopes one connection receives.
struct ConnectionInbound {
    plugin_name: String,
    generation: u64,
    authenticated_peer: bool,
    pending: PendingResponses,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
    outbound_tx: mpsc::Sender<super::proto::Envelope>,
    binding_capacity: Arc<tokio::sync::Semaphore>,
    control_admission: rpc_admission::RpcAdmission,
    bridge_admission: rpc_admission::RpcAdmission,
    observation: crate::plugin::inference_observation::subscription::ConnectionSubscription,
    // Started by the first status publication, if any.
    status_lane: Option<crate::plugin::runtime::status_publication::StatusLane>,
}

/// Reads frames until the plugin closes the stream or sends a bad frame.
///
/// The frame read is not cancel-safe, so this future must stay alive until
/// the connection ends; it is never restarted between frames.
async fn read_inbound<R>(reader: &mut R, inbound: &mut ConnectionInbound) -> Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
{
    loop {
        let envelope = frames::read_frame(
            reader,
            frames::PREFIX_READ_TIMEOUT,
            frames::BODY_READ_TIMEOUT,
        )
        .await?;
        inbound.dispatch(envelope).await?;
    }
}

/// Writes queued envelopes in order until the queue closes or a write fails.
async fn write_outbound<W>(
    writer: &mut W,
    outbound_rx: &mut mpsc::Receiver<super::proto::Envelope>,
) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    while let Some(envelope) = outbound_rx.recv().await {
        frames::write_frame(writer, &envelope).await?;
    }
    Ok(())
}

impl ConnectionInbound {
    fn request_peer(&self) -> PluginRequestPeer {
        PluginRequestPeer {
            authenticated: self.authenticated_peer,
            runtime: self.runtime.clone(),
            generation: self.generation,
        }
    }

    /// A slot in `pool` for a request the plugin started, `None` once it was
    /// refused, or an error that ends the connection.
    fn admit(
        &self,
        pool: &rpc_admission::RpcAdmission,
        request_id: u64,
    ) -> Result<Option<rpc_admission::Admitted>> {
        pool.admit(&self.plugin_name, request_id, &self.outbound_tx)
    }

    async fn dispatch(&mut self, envelope: super::proto::Envelope) -> Result<()> {
        use super::proto::envelope::Payload;

        let request_id = envelope.request_id;
        match envelope.payload {
            Some(Payload::ChannelMessage(message)) => {
                let plugin_id = sender_plugin_id(&self.plugin_name, &envelope.plugin_id);
                let _ = self
                    .mesh_tx
                    .send(PluginMeshEvent::Channel { plugin_id, message })
                    .await;
            }
            Some(Payload::BulkTransferMessage(message)) => {
                let plugin_id = sender_plugin_id(&self.plugin_name, &envelope.plugin_id);
                let _ = self
                    .mesh_tx
                    .send(PluginMeshEvent::BulkTransfer { plugin_id, message })
                    .await;
            }
            Some(Payload::PublishPluginStatusRequest(request)) => {
                // Shown under the connection's own name,
                // so one plugin cannot publish as another.
                self.status_lane
                    .get_or_insert_with(|| {
                        crate::plugin::runtime::status_publication::StatusLane::spawn(
                            self.plugin_name.clone(),
                            self.generation,
                            self.runtime.clone(),
                            self.mesh_tx.clone(),
                            self.outbound_tx.clone(),
                        )
                    })
                    .push(request_id, request);
            }
            Some(Payload::SubscribeInferenceObservationRequest(_)) => {
                crate::plugin::inference_observation::subscription::spawn(
                    crate::plugin::inference_observation::hub(),
                    crate::plugin::inference_observation::subscription::SubscribeRequest {
                        plugin_name: self.plugin_name.clone(),
                        generation: self.generation,
                        request_id,
                        runtime: self.runtime.clone(),
                        mesh_tx: self.mesh_tx.clone(),
                        outbound_tx: self.outbound_tx.clone(),
                        subscription: self.observation.clone(),
                    },
                );
            }
            Some(Payload::RpcRequest(request))
                if request.method == mesh_llm_plugin::operation_binding::METHOD =>
            {
                // Bounded by its own, smaller binding capacity.
                operation_binding::forward(
                    self.plugin_name.clone(),
                    request_id,
                    request,
                    self.request_peer(),
                    self.mesh_tx.clone(),
                    self.outbound_tx.clone(),
                    self.binding_capacity.clone(),
                );
            }
            Some(Payload::RpcNotification(notification)) => {
                forward_plugin_notification(
                    self.plugin_name.clone(),
                    notification,
                    self.rpc_bridge.clone(),
                );
            }
            Some(payload) => {
                if let Some(response) =
                    self.forward_request(request_id, &envelope.plugin_id, payload)?
                {
                    self.settle_pending(
                        request_id,
                        envelope.protocol_version,
                        envelope.plugin_id,
                        Some(response),
                    )
                    .await;
                }
            }
            None => {
                self.settle_pending(
                    request_id,
                    envelope.protocol_version,
                    envelope.plugin_id,
                    None,
                )
                .await;
            }
        }
        Ok(())
    }

    /// Hands a request the plugin started to its handler task, once it has a
    /// slot. Any other payload is given back: it answers a host request.
    fn forward_request(
        &self,
        request_id: u64,
        claimed_plugin_id: &str,
        payload: super::proto::envelope::Payload,
    ) -> Result<Option<super::proto::envelope::Payload>> {
        use super::proto::envelope::Payload;

        let plugin_name = self.plugin_name.clone();
        let mesh_tx = self.mesh_tx.clone();
        let outbound_tx = self.outbound_tx.clone();
        match payload {
            Payload::OpenMeshStreamRequest(request) => {
                if let Some(admitted) = self.admit(&self.control_admission, request_id)? {
                    forward_plugin_mesh_stream_request(
                        sender_plugin_id(&plugin_name, claimed_plugin_id),
                        admitted,
                        request,
                        mesh_tx,
                        outbound_tx,
                    );
                }
            }
            Payload::PrepareModelAssetsRequest(request) => {
                if let Some(admitted) = self.admit(&self.control_admission, request_id)? {
                    crate::plugin::model_assets::prepare_for_connection(
                        admitted.request_id,
                        request,
                        plugin_name,
                        self.generation,
                        self.runtime.clone(),
                        outbound_tx,
                        admitted.permit,
                    );
                }
            }
            // The requester is the connection's own name,
            // so one plugin cannot act as another.
            Payload::PeerBlockRequest(request) => {
                if let Some(admitted) = self.admit(&self.control_admission, request_id)? {
                    forward_plugin_peer_block_request(
                        plugin_name,
                        admitted,
                        request,
                        mesh_tx,
                        outbound_tx,
                    );
                }
            }
            Payload::ModelRequirementRequest(request) => {
                if let Some(admitted) = self.admit(&self.control_admission, request_id)? {
                    model_requirements::forward_request(
                        plugin_name,
                        admitted,
                        request,
                        mesh_tx,
                        outbound_tx,
                    );
                }
            }
            // Announced under the connection's own name,
            // so one plugin cannot set another's key.
            Payload::PluginKeyRequest(request) => {
                if let Some(admitted) = self.admit(&self.control_admission, request_id)? {
                    forward_plugin_key_request(
                        plugin_name,
                        admitted,
                        request,
                        mesh_tx,
                        outbound_tx,
                    );
                }
            }
            Payload::RpcRequest(request) => {
                if let Some(admitted) = self.admit(&self.bridge_admission, request_id)? {
                    forward_plugin_request(
                        plugin_name,
                        admitted,
                        request,
                        mesh_tx,
                        self.rpc_bridge.clone(),
                        outbound_tx,
                        self.request_peer(),
                    );
                }
            }
            response => return Ok(Some(response)),
        }
        Ok(None)
    }

    /// Delivers a response to the host request that is waiting for it.
    async fn settle_pending(
        &self,
        request_id: u64,
        protocol_version: u32,
        plugin_id: String,
        payload: Option<super::proto::envelope::Payload>,
    ) {
        let responder = self.pending.lock().await.remove(&request_id);
        if let Some(responder) = responder {
            let _ = responder.send(Ok(super::proto::Envelope {
                protocol_version,
                plugin_id,
                request_id,
                payload,
            }));
        } else {
            tracing::debug!(
                "Plugin '{}' sent an unsolicited response id={}",
                self.plugin_name,
                request_id
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::PluginWebUiState;
    use tokio::io::DuplexStream;

    const PLUGIN: &str = "demo";
    // Generous so a busy machine cannot turn a correct run into a failure.
    const WAIT: Duration = Duration::from_secs(30);

    struct Harness {
        plugin: DuplexStream,
        outbound_tx: mpsc::Sender<proto::Envelope>,
        pending: PendingResponses,
        mesh_rx: mpsc::Receiver<PluginMeshEvent>,
        _connection: tokio::task::JoinHandle<()>,
    }

    fn summary() -> PluginSummary {
        PluginSummary {
            name: PLUGIN.into(),
            kind: "external".into(),
            enabled: true,
            status: "running".into(),
            pid: None,
            version: None,
            capabilities: Vec::new(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            web_ui: PluginWebUiState::default(),
            startup: None,
            error: None,
        }
    }

    /// A host connection over an in-memory pipe that buffers `pipe_bytes`
    /// in each direction; the test plays the plugin on the other end.
    fn connect(pipe_bytes: usize) -> Harness {
        let (host, plugin) = tokio::io::duplex(pipe_bytes);
        let (outbound_tx, outbound_rx) = mpsc::channel(256);
        let (mesh_tx, mesh_rx) = mpsc::channel(256);
        let pending: PendingResponses = Arc::new(Mutex::new(HashMap::new()));
        let connection = tokio::spawn(connection_loop(
            LocalStream::Memory(host),
            outbound_rx,
            pending.clone(),
            mesh_tx,
            PLUGIN.into(),
            Arc::new(Mutex::new(summary())),
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
            outbound_tx.clone(),
            1,
            Arc::new(AtomicBool::new(false)),
        ));
        Harness {
            plugin,
            outbound_tx,
            pending,
            mesh_rx,
            _connection: connection,
        }
    }

    fn envelope(request_id: u64, payload: proto::envelope::Payload) -> proto::Envelope {
        proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: PLUGIN.into(),
            request_id,
            payload: Some(payload),
        }
    }

    fn rpc_response(result_json: String) -> proto::envelope::Payload {
        proto::envelope::Payload::RpcResponse(proto::RpcResponse { result_json })
    }

    fn encoded(envelope: &proto::Envelope) -> Vec<u8> {
        let body = prost::Message::encode_to_vec(envelope);
        let mut frame = (body.len() as u32).to_le_bytes().to_vec();
        frame.extend_from_slice(&body);
        frame
    }

    async fn expect_host_request(
        pending: &PendingResponses,
        request_id: u64,
    ) -> oneshot::Receiver<Result<proto::Envelope>> {
        let (response_tx, response_rx) = oneshot::channel();
        pending.lock().await.insert(request_id, response_tx);
        response_rx
    }

    async fn read_from_host(plugin: &mut DuplexStream) -> proto::Envelope {
        frames::read_frame(plugin, WAIT, WAIT)
            .await
            .expect("the host sends a whole frame")
    }

    async fn settled(response_rx: oneshot::Receiver<Result<proto::Envelope>>) -> proto::Envelope {
        tokio::time::timeout(WAIT, response_rx)
            .await
            .expect("the host delivers the plugin's response")
            .expect("the responder is kept until the response arrives")
            .expect("the connection stays up")
    }

    fn result_json(envelope: proto::Envelope) -> String {
        let Some(proto::envelope::Payload::RpcResponse(response)) = envelope.payload else {
            panic!("expected an RPC response");
        };
        response.result_json
    }

    #[tokio::test]
    async fn an_outbound_send_does_not_lose_a_partly_read_inbound_frame() {
        let mut harness = connect(64 * 1024);
        let response_rx = expect_host_request(&harness.pending, 42).await;
        let body = "x".repeat(8 * 1024);
        let frame = encoded(&envelope(42, rpc_response(body.clone())));
        let (head, tail) = frame.split_at(frame.len() / 2);

        harness.plugin.write_all(head).await.unwrap();
        // Let the host take the first half of the frame before it has
        // something to send. If it has not yet, the test can only pass without
        // proving anything; it cannot fail a correct host.
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
        harness
            .outbound_tx
            .send(envelope(
                7,
                proto::envelope::Payload::HealthRequest(proto::HealthRequest {}),
            ))
            .await
            .unwrap();
        assert_eq!(read_from_host(&mut harness.plugin).await.request_id, 7);
        harness.plugin.write_all(tail).await.unwrap();

        assert_eq!(result_json(settled(response_rx).await), body);
    }

    #[tokio::test]
    async fn a_blocked_outbound_write_does_not_stop_inbound_reads() {
        // The pipe holds far less than one outbound frame, and the plugin
        // reads nothing until its own response is through.
        let mut harness = connect(64);
        let request = "r".repeat(16 * 1024);
        harness
            .outbound_tx
            .send(envelope(
                7,
                proto::envelope::Payload::RpcRequest(proto::RpcRequest {
                    invocation_id: None,
                    method: "demo/echo".into(),
                    params_json: request.clone(),
                }),
            ))
            .await
            .unwrap();
        let response_rx = expect_host_request(&harness.pending, 42).await;
        let response = "y".repeat(4 * 1024);
        let frame = encoded(&envelope(42, rpc_response(response.clone())));

        tokio::time::timeout(WAIT, harness.plugin.write_all(&frame))
            .await
            .expect("the host keeps reading while its write waits")
            .unwrap();
        assert_eq!(result_json(settled(response_rx).await), response);

        let outbound = read_from_host(&mut harness.plugin).await;
        let Some(proto::envelope::Payload::RpcRequest(sent)) = outbound.payload else {
            panic!("expected the queued RPC request");
        };
        assert_eq!(sent.params_json, request);
    }

    async fn send_peer_block_request(plugin: &mut DuplexStream, request_id: u64) {
        let request = envelope(
            request_id,
            proto::envelope::Payload::PeerBlockRequest(proto::PeerBlockRequest::default()),
        );
        frames::write_frame(plugin, &request).await.unwrap();
    }

    async fn next_peer_block(
        mesh_rx: &mut mpsc::Receiver<PluginMeshEvent>,
    ) -> oneshot::Sender<Result<proto::PeerBlockResponse, proto::ErrorResponse>> {
        let event = tokio::time::timeout(WAIT, mesh_rx.recv())
            .await
            .expect("the host forwards an admitted request")
            .expect("the mesh queue stays open");
        let PluginMeshEvent::PeerBlock { response_tx, .. } = event else {
            panic!("expected a peer block request");
        };
        response_tx
    }

    #[tokio::test]
    async fn control_requests_over_the_bound_are_refused_until_one_finishes() {
        let capacity = rpc_admission::CONTROL_REQUEST_CAPACITY as u64;
        let mut harness = connect(64 * 1024);
        let mut in_flight = Vec::new();
        for request_id in 1..=capacity {
            send_peer_block_request(&mut harness.plugin, request_id).await;
            in_flight.push(next_peer_block(&mut harness.mesh_rx).await);
        }

        send_peer_block_request(&mut harness.plugin, capacity + 1).await;
        let refusal = read_from_host(&mut harness.plugin).await;
        assert_eq!(refusal.request_id, capacity + 1);
        let Some(proto::envelope::Payload::ErrorResponse(error)) = refusal.payload else {
            panic!("a request over the bound is answered with an error");
        };
        assert!(error.message.contains("too many host requests"));
        assert!(harness.mesh_rx.try_recv().is_err());

        // Bridge RPCs have a pool of their own, so they are still served.
        let rpc = envelope(
            capacity + 3,
            proto::envelope::Payload::RpcRequest(proto::RpcRequest {
                invocation_id: None,
                method: "demo/echo".into(),
                params_json: "{}".into(),
            }),
        );
        frames::write_frame(&mut harness.plugin, &rpc)
            .await
            .unwrap();
        let reply = read_from_host(&mut harness.plugin).await;
        assert_eq!(reply.request_id, capacity + 3);
        let Some(proto::envelope::Payload::ErrorResponse(error)) = reply.payload else {
            panic!("without a bridge the RPC is answered with an error");
        };
        assert!(error.message.contains("No active MCP bridge"));

        let finished = in_flight.remove(0);
        finished
            .send(Ok(proto::PeerBlockResponse::default()))
            .unwrap();
        let answered = read_from_host(&mut harness.plugin).await;
        assert_eq!(answered.request_id, 1);
        assert!(matches!(
            answered.payload,
            Some(proto::envelope::Payload::PeerBlockResponse(_))
        ));

        send_peer_block_request(&mut harness.plugin, capacity + 2).await;
        drop(next_peer_block(&mut harness.mesh_rx).await);
    }
}
