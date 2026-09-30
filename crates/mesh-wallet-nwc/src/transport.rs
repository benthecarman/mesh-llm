//! NIP-47 over Nostr relays.
//!
//! One subscription, opened before the first request, receives every response
//! and notification the wallet service addresses to this client. Requests are
//! matched to responses by the request event id, so a response can never be
//! missed between publishing a request and listening for its answer.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use nostr_sdk::prelude::*;
use serde_json::Value;
use tokio::sync::{broadcast, oneshot};

use crate::wire::{self, Notification, ResponseError, WalletServiceError};

const INFO_KIND: u16 = 13194;
const REQUEST_KIND: u16 = 23194;
const RESPONSE_KIND: u16 = 23195;
const NIP04_NOTIFICATION_KIND: u16 = 23196;
const NIP44_NOTIFICATION_KIND: u16 = 23197;

/// Why a request produced no result.
#[derive(Debug)]
pub enum TransportError {
    /// The request was never handed to a relay, so the wallet cannot have
    /// acted on it.
    NotSent(String),
    /// The request was handed to at least one relay but no answer arrived.
    /// The wallet may still have acted on it.
    NoResponse(String),
    /// The wallet answered with an error.
    Wallet(WalletServiceError),
    /// The wallet answered with something this plugin cannot read.
    Malformed(String),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSent(reason) => write!(f, "request not sent: {reason}"),
            Self::NoResponse(reason) => write!(f, "no response from wallet: {reason}"),
            Self::Wallet(error) => write!(f, "wallet error: {error}"),
            Self::Malformed(reason) => write!(f, "malformed wallet response: {reason}"),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<ResponseError> for TransportError {
    fn from(error: ResponseError) -> Self {
        match error {
            ResponseError::Wallet(error) => Self::Wallet(error),
            ResponseError::Malformed(reason) => Self::Malformed(reason),
        }
    }
}

/// How the provider talks to a wallet service. Implemented over relays in
/// production and in memory in tests.
#[async_trait]
pub trait Transport: Send + Sync + 'static {
    /// Send one request and wait up to `timeout` for its result.
    async fn request(
        &self,
        method: &'static str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, TransportError>;

    /// Payment notifications from now on. Delivery is best effort.
    fn notifications(&self) -> broadcast::Receiver<Notification>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cipher {
    Nip44,
    Nip04,
}

impl Cipher {
    /// The wallet's info event lists the ciphers it accepts. Wallets that
    /// predate the `encryption` tag only speak NIP-04.
    fn from_info(info: Option<&Event>) -> Self {
        let advertised = info.and_then(|event| {
            event
                .tags
                .iter()
                .find(|tag| tag.kind() == "encryption")
                .and_then(|tag| tag.content())
                .map(str::to_owned)
        });
        match advertised {
            Some(ciphers) if ciphers.split_whitespace().any(|c| c == "nip44_v2") => Self::Nip44,
            _ => Self::Nip04,
        }
    }

    fn encrypt(self, keys: &Keys, peer: &PublicKey, content: &str) -> Result<String> {
        Ok(match self {
            Self::Nip44 => nip44::encrypt(keys.secret_key(), peer, content, nip44::Version::V2)?,
            Self::Nip04 => nip04::encrypt(keys.secret_key(), peer, content)?,
        })
    }
}

/// Decrypt a wallet event, recognizing the cipher from the payload shape:
/// NIP-04 payloads carry an `?iv=` suffix, NIP-44 payloads never do.
fn decrypt(keys: &Keys, peer: &PublicKey, content: &str) -> Result<String> {
    Ok(if content.contains("?iv=") {
        nip04::decrypt(keys.secret_key(), peer, content)?
    } else {
        nip44::decrypt(keys.secret_key(), peer, content)?
    })
}

type Pending = Arc<Mutex<HashMap<EventId, oneshot::Sender<Event>>>>;

pub struct RelayTransport {
    client: Client,
    keys: Keys,
    wallet: PublicKey,
    cipher: Cipher,
    /// The methods the wallet's info event lists.
    advertised_methods: Vec<String>,
    pending: Pending,
    notifications: broadcast::Sender<Notification>,
    router: tokio::task::JoinHandle<()>,
}

impl RelayTransport {
    /// Connect to the URI's relays, learn the wallet's cipher and start
    /// listening for responses and notifications.
    pub async fn connect(uri: &NostrWalletConnectUri, timeout: Duration) -> Result<Self> {
        // The relay websockets use rustls; this process has no other TLS user
        // to install a provider first.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let keys = Keys::new(uri.secret.clone());
        let client = Client::new();
        for relay in &uri.relays {
            client
                .add_relay(relay)
                .await
                .with_context(|| format!("add relay {relay}"))?;
        }
        client.connect().and_wait(timeout).await;
        if !any_relay_connected(&client).await {
            anyhow::bail!("could not connect to any wallet relay within {timeout:?}");
        }

        let info = client
            .fetch_events(
                Filter::new()
                    .kind(Kind::from_u16(INFO_KIND))
                    .author(uri.public_key)
                    .limit(1),
            )
            .timeout(timeout)
            .await
            .context("fetch wallet info event")?;
        let cipher = Cipher::from_info(info.first());
        let advertised_methods = info
            .first()
            .map(|event| {
                event
                    .content
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();

        let pending: Pending = Arc::default();
        let (notifications, _) = broadcast::channel(256);
        // Listen before subscribing so nothing the subscription delivers is
        // missed.
        let events = client.notifications();
        client
            .subscribe(
                Filter::new()
                    .kinds([
                        Kind::from_u16(RESPONSE_KIND),
                        Kind::from_u16(NIP04_NOTIFICATION_KIND),
                        Kind::from_u16(NIP44_NOTIFICATION_KIND),
                    ])
                    .author(uri.public_key)
                    .pubkey(keys.public_key())
                    .since(Timestamp::now()),
            )
            .await
            .context("subscribe to wallet responses")?;
        let router = tokio::spawn(route(
            events,
            keys.clone(),
            uri.public_key,
            Arc::clone(&pending),
            notifications.clone(),
        ));
        Ok(Self {
            client,
            keys,
            wallet: uri.public_key,
            cipher,
            advertised_methods,
            pending,
            notifications,
            router,
        })
    }

    /// The methods the wallet's info event lists: what the wallet supports,
    /// not necessarily what this connection may call.
    pub fn advertised_methods(&self) -> &[String] {
        &self.advertised_methods
    }

    fn request_event(&self, method: &str, params: Value, timeout: Duration) -> Result<Event> {
        let body = serde_json::to_string(&serde_json::json!({
            "method": method,
            "params": params,
        }))?;
        let mut tags = vec![
            Tag::public_key(self.wallet),
            // A request that reaches the wallet late must not be executed
            // after this plugin has stopped waiting for it.
            Tag::expiration(Timestamp::now() + timeout),
        ];
        if self.cipher == Cipher::Nip44 {
            tags.push(Tag::custom("encryption", ["nip44_v2"]));
        }
        let content = self.cipher.encrypt(&self.keys, &self.wallet, &body)?;
        Ok(EventBuilder::new(Kind::from_u16(REQUEST_KIND), content)
            .tags(tags)
            .finalize(&self.keys)?)
    }

    fn decode_response(&self, method: &str, event: &Event) -> Result<Value, TransportError> {
        let body = decrypt(&self.keys, &self.wallet, &event.content)
            .map_err(|error| TransportError::Malformed(format!("decrypt response: {error}")))?;
        let response: wire::Response = serde_json::from_str(&body)
            .map_err(|error| TransportError::Malformed(format!("decode response: {error}")))?;
        Ok(wire::into_result(response, method)?)
    }
}

impl Drop for RelayTransport {
    fn drop(&mut self) {
        self.router.abort();
    }
}

#[async_trait]
impl Transport for RelayTransport {
    async fn request(
        &self,
        method: &'static str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, TransportError> {
        let event = self
            .request_event(method, params, timeout)
            .map_err(|error| TransportError::NotSent(format!("build request: {error:#}")))?;
        // Only a request that never reached a relay is provably unsent: a
        // relay that failed to acknowledge may still have forwarded it.
        if !any_relay_connected(&self.client).await {
            return Err(TransportError::NotSent(
                "no wallet relay is connected".into(),
            ));
        }
        let (tx, rx) = oneshot::channel();
        lock(&self.pending).insert(event.id, tx);
        let _pending = PendingGuard {
            pending: &self.pending,
            id: event.id,
        };
        if let Err(error) = self.client.send_event(&event).await {
            return Err(TransportError::NoResponse(format!(
                "publish request: {error}"
            )));
        }
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(response)) => self.decode_response(method, &response),
            Ok(Err(_)) => Err(TransportError::NoResponse("wallet listener stopped".into())),
            Err(_) => Err(TransportError::NoResponse(format!(
                "no {method} response within {timeout:?}"
            ))),
        }
    }

    fn notifications(&self) -> broadcast::Receiver<Notification> {
        self.notifications.subscribe()
    }
}

/// Removes a request's response slot however the request ends.
struct PendingGuard<'a> {
    pending: &'a Pending,
    id: EventId,
}

impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {
        lock(self.pending).remove(&self.id);
    }
}

fn lock(pending: &Pending) -> std::sync::MutexGuard<'_, HashMap<EventId, oneshot::Sender<Event>>> {
    pending
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

async fn any_relay_connected(client: &Client) -> bool {
    client
        .relays()
        .await
        .values()
        .any(|relay| relay.status() == RelayStatus::Connected)
}

/// Deliver responses to their waiting request and decode notifications.
async fn route(
    mut events: std::pin::Pin<Box<dyn futures::Stream<Item = ClientNotification> + Send>>,
    keys: Keys,
    wallet: PublicKey,
    pending: Pending,
    notifications: broadcast::Sender<Notification>,
) {
    while let Some(notification) = events.next().await {
        let ClientNotification::Event { event, .. } = notification else {
            continue;
        };
        if event.pubkey != wallet {
            continue;
        }
        match event.kind.as_u16() {
            RESPONSE_KIND => {
                let Some(request) = event.tags.event_ids().next() else {
                    continue;
                };
                if let Some(waiter) = lock(&pending).remove(&request) {
                    let _ = waiter.send(*event);
                }
            }
            NIP04_NOTIFICATION_KIND | NIP44_NOTIFICATION_KIND => {
                let decoded = decrypt(&keys, &wallet, &event.content)
                    .ok()
                    .and_then(|body| serde_json::from_str::<Notification>(&body).ok());
                if let Some(decoded) = decoded {
                    let _ = notifications.send(decoded);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
#[path = "transport_tests.rs"]
mod relay_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn info_with(tag: Option<&str>) -> Event {
        let keys = Keys::generate();
        let mut builder = EventBuilder::new(Kind::from_u16(INFO_KIND), "pay_invoice");
        if let Some(value) = tag {
            builder = builder.tag(Tag::custom("encryption", [value]));
        }
        builder.finalize(&keys).unwrap()
    }

    #[test]
    fn cipher_follows_the_wallet_info_event() {
        assert_eq!(Cipher::from_info(None), Cipher::Nip04);
        assert_eq!(Cipher::from_info(Some(&info_with(None))), Cipher::Nip04);
        assert_eq!(
            Cipher::from_info(Some(&info_with(Some("nip04")))),
            Cipher::Nip04
        );
        assert_eq!(
            Cipher::from_info(Some(&info_with(Some("nip44_v2 nip04")))),
            Cipher::Nip44
        );
    }

    #[test]
    fn both_ciphers_round_trip_between_client_and_wallet() {
        let client = Keys::generate();
        let wallet = Keys::generate();
        for cipher in [Cipher::Nip44, Cipher::Nip04] {
            let sealed = cipher
                .encrypt(&client, &wallet.public_key(), r#"{"method":"get_balance"}"#)
                .unwrap();
            let opened = decrypt(&wallet, &client.public_key(), &sealed).unwrap();
            assert_eq!(opened, r#"{"method":"get_balance"}"#, "{cipher:?}");
        }
    }
}
