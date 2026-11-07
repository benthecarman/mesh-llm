// A legacy raw tunnel stream (0x02) is refused on its own: the connection
// keeps dispatching the peer's other streams afterwards.

/// Ask `receiver` for its routing table over `conn` and decode the answer.
async fn request_route_table_over(
    conn: &Connection,
    requester: &Node,
) -> crate::proto::node::RouteTable {
    use prost::Message as _;
    let (mut send, mut recv) = conn.open_bi().await.expect("open route request stream");
    let request = crate::proto::node::RouteTableRequest {
        requester_id: requester.id().as_bytes().to_vec(),
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    send.write_all(&encode_control_frame(STREAM_ROUTE_REQUEST, &request))
        .await
        .expect("write route request");
    let body = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        read_len_prefixed(&mut recv),
    )
    .await
    .expect("the receiver answers the route request")
    .expect("read route table");
    crate::proto::node::RouteTable::decode(body.as_slice()).expect("decode route table")
}

#[tokio::test]
async fn legacy_tunnel_stream_is_refused_without_ending_stream_dispatch() {
    let (sender, receiver) = connect_plugin_identity_pair().await;
    // The receiver must have admitted the sender, or the raw tunnel stream
    // would be dropped at the admission gate before it reached dispatch.
    let sender_id = sender.id();
    wait_until("the receiver has admitted the sender", || async {
        receiver
            .state
            .lock()
            .await
            .peers
            .get(&sender_id)
            .is_some_and(PeerInfo::is_admitted)
    })
    .await;
    let conn = sender
        .connection_to_peer(receiver.id())
        .await
        .expect("authenticated connection to receiver");

    let (mut send, mut recv) = conn.open_bi().await.expect("open legacy tunnel stream");
    send.write_all(&[STREAM_TUNNEL]).await.unwrap();
    send.write_all(b"\x00\x01\x02\x03RPC-BYTES").await.unwrap();
    let ended = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        let mut buf = [0u8; 64];
        loop {
            match recv.read(&mut buf).await {
                Ok(None) | Err(_) => return,
                Ok(Some(_)) => {}
            }
        }
    })
    .await;
    assert!(
        ended.is_ok(),
        "the receiver must refuse the legacy tunnel stream"
    );

    // The connection still serves the peer's next stream.
    request_route_table_over(&conn, &sender).await;
}
