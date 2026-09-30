//! `RelayTransport` against an in-process relay and a scripted NIP-47 wallet
//! service, so event kinds, tags, ciphers and response routing are exercised
//! end to end.

use nostr_sdk::prelude::MockRelay;
use serde_json::json;

use super::*;

/// A minimal wallet service: answers `get_balance`, refuses everything else
/// with `NOT_IMPLEMENTED`, and replies in the cipher the request used.
async fn run_wallet_service(relay: RelayUrl, wallet: Keys, speaks_nip44: bool) -> Client {
    let client = Client::new();
    client.add_relay(&relay).await.unwrap();
    client.connect().and_wait(Duration::from_secs(5)).await;
    let mut info = EventBuilder::new(Kind::from_u16(INFO_KIND), "get_balance pay_invoice");
    if speaks_nip44 {
        info = info.tag(Tag::custom("encryption", ["nip44_v2 nip04"]));
    }
    client
        .send_event(&info.finalize(&wallet).unwrap())
        .await
        .unwrap();

    let mut events = client.notifications();
    client
        .subscribe(
            Filter::new()
                .kind(Kind::from_u16(REQUEST_KIND))
                .pubkey(wallet.public_key()),
        )
        .await
        .unwrap();
    let responder = client.clone();
    tokio::spawn(async move {
        while let Some(notification) = events.next().await {
            let ClientNotification::Event { event, .. } = notification else {
                continue;
            };
            if event.kind.as_u16() != REQUEST_KIND {
                continue;
            }
            let nip04 = event.content.contains("?iv=");
            let body = decrypt(&wallet, &event.pubkey, &event.content).unwrap();
            let request: Value = serde_json::from_str(&body).unwrap();
            let method = request["method"].as_str().unwrap().to_owned();
            let response = if method == "get_balance" {
                json!({"result_type": method, "result": {"balance": 21_000}})
            } else {
                json!({
                    "result_type": method,
                    "error": {"code": "NOT_IMPLEMENTED", "message": "nope"}
                })
            };
            let cipher = if nip04 { Cipher::Nip04 } else { Cipher::Nip44 };
            let sealed = cipher
                .encrypt(&wallet, &event.pubkey, &response.to_string())
                .unwrap();
            let reply = EventBuilder::new(Kind::from_u16(RESPONSE_KIND), sealed)
                .tags([Tag::public_key(event.pubkey), Tag::event(event.id)])
                .finalize(&wallet)
                .unwrap();
            responder.send_event(&reply).await.unwrap();

            let notification = json!({
                "notification_type": "payment_received",
                "notification": {"type": "incoming", "payment_hash": "aa", "amount": 1}
            });
            let sealed = cipher
                .encrypt(&wallet, &event.pubkey, &notification.to_string())
                .unwrap();
            let kind = if nip04 {
                NIP04_NOTIFICATION_KIND
            } else {
                NIP44_NOTIFICATION_KIND
            };
            let note = EventBuilder::new(Kind::from_u16(kind), sealed)
                .tag(Tag::public_key(event.pubkey))
                .finalize(&wallet)
                .unwrap();
            responder.send_event(&note).await.unwrap();
        }
    });
    client
}

fn uri(relay: &RelayUrl, wallet: &Keys) -> NostrWalletConnectUri {
    NostrWalletConnectUri::new(
        wallet.public_key(),
        vec![relay.clone()],
        Keys::generate().secret_key().clone(),
        None,
    )
}

async fn round_trip(speaks_nip44: bool) {
    let relay = MockRelay::run().await.unwrap();
    let url = relay.url().await;
    let wallet = Keys::generate();
    let _service = run_wallet_service(url.clone(), wallet.clone(), speaks_nip44).await;

    let transport = RelayTransport::connect(&uri(&url, &wallet), Duration::from_secs(5))
        .await
        .unwrap();
    let expected = if speaks_nip44 {
        Cipher::Nip44
    } else {
        Cipher::Nip04
    };
    assert_eq!(transport.cipher, expected);
    assert_eq!(
        transport.advertised_methods(),
        ["get_balance", "pay_invoice"]
    );

    let mut notifications = transport.notifications();
    let balance = transport
        .request("get_balance", json!({}), Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(balance, json!({"balance": 21_000}));
    let note = tokio::time::timeout(Duration::from_secs(5), notifications.recv())
        .await
        .expect("notification must arrive")
        .unwrap();
    assert_eq!(note.notification.payment_hash, "aa");

    match transport
        .request(
            "pay_invoice",
            json!({"invoice": "lnbc"}),
            Duration::from_secs(5),
        )
        .await
    {
        Err(TransportError::Wallet(error)) => assert_eq!(error.code, "NOT_IMPLEMENTED"),
        other => panic!("expected a wallet error, got {other:?}"),
    }
}

#[tokio::test]
async fn nip44_wallet_round_trips_requests_and_notifications() {
    round_trip(true).await;
}

#[tokio::test]
async fn legacy_nip04_wallet_round_trips_requests_and_notifications() {
    round_trip(false).await;
}

#[tokio::test]
async fn unanswered_request_is_uncertain_and_an_unreachable_relay_is_unsent() {
    let relay = MockRelay::run().await.unwrap();
    let url = relay.url().await;
    // No wallet service is listening.
    let wallet = Keys::generate();
    let transport = RelayTransport::connect(&uri(&url, &wallet), Duration::from_secs(2))
        .await
        .unwrap();
    let silent = transport
        .request("get_balance", json!({}), Duration::from_millis(300))
        .await;
    assert!(
        matches!(silent, Err(TransportError::NoResponse(_))),
        "{silent:?}"
    );

    relay.shutdown();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let unsent = loop {
        let result = transport
            .request("get_balance", json!({}), Duration::from_millis(200))
            .await;
        if matches!(result, Err(TransportError::NotSent(_)))
            || tokio::time::Instant::now() > deadline
        {
            break result;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    assert!(
        matches!(unsent, Err(TransportError::NotSent(_))),
        "{unsent:?}"
    );
}
