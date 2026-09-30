//! The provider against a scripted in-memory wallet service, so every
//! money-moving rule is checked without relays.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use bitcoin::hashes::{Hash, sha256};
use bitcoin::secp256k1::{Secp256k1, SecretKey};
use lightning_invoice::{Currency, InvoiceBuilder, PaymentHash, PaymentSecret};
use serde_json::json;

use super::*;
use crate::wire::{WalletMethods, WalletServiceError};

type Reply = Result<Value, TransportError>;

struct FakeWallet {
    replies: Mutex<HashMap<&'static str, VecDeque<Reply>>>,
    calls: Mutex<Vec<(&'static str, Value)>>,
    notifications: broadcast::Sender<Notification>,
}

impl FakeWallet {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::default(),
            calls: Mutex::default(),
            notifications: broadcast::channel(16).0,
        })
    }

    fn reply(&self, method: &'static str, reply: Reply) {
        self.replies
            .lock()
            .unwrap()
            .entry(method)
            .or_default()
            .push_back(reply);
    }

    fn methods(&self) -> Vec<&'static str> {
        self.calls.lock().unwrap().iter().map(|(m, _)| *m).collect()
    }

    fn params(&self, method: &str) -> Value {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|(m, _)| *m == method)
            .map(|(_, params)| params.clone())
            .unwrap()
    }
}

#[async_trait]
impl Transport for FakeWallet {
    async fn request(
        &self,
        method: &'static str,
        params: Value,
        _timeout: Duration,
    ) -> Result<Value, TransportError> {
        self.calls.lock().unwrap().push((method, params));
        self.replies
            .lock()
            .unwrap()
            .get_mut(method)
            .and_then(VecDeque::pop_front)
            .unwrap_or_else(|| Err(TransportError::NoResponse(format!("unscripted {method}"))))
    }

    fn notifications(&self) -> broadcast::Receiver<Notification> {
        self.notifications.subscribe()
    }
}

fn provider(wallet: &Arc<FakeWallet>) -> NwcProvider {
    provider_with(wallet, WalletMethods::default())
}

/// A provider for a wallet that offers NWC-321 `pay` and `receive`.
fn nwc321_provider(wallet: &Arc<FakeWallet>) -> NwcProvider {
    provider_with(
        wallet,
        WalletMethods {
            pay: true,
            receive: true,
        },
    )
}

fn provider_with(wallet: &Arc<FakeWallet>, methods: WalletMethods) -> NwcProvider {
    NwcProvider::new(
        Arc::clone(wallet) as Arc<dyn Transport>,
        Timing {
            request_timeout: Duration::from_secs(1),
            pay_timeout: Duration::from_secs(1),
            // Long enough that a test only finishes if a notification wakes it.
            poll_interval: Duration::from_secs(60),
        },
        methods,
    )
}

const PREIMAGE: [u8; 32] = [5; 32];

fn signed_invoice(amount_msat: Option<u64>, expiry_secs: u64) -> String {
    let secret = SecretKey::from_slice(&[7; 32]).unwrap();
    let hash = sha256::Hash::hash(&PREIMAGE).to_byte_array();
    let mut builder = InvoiceBuilder::new(Currency::Bitcoin)
        .description("test".into())
        .payment_hash(PaymentHash(hash))
        .payment_secret(PaymentSecret([42; 32]))
        .current_timestamp()
        .expiry_time(Duration::from_secs(expiry_secs))
        .min_final_cltv_expiry_delta(144);
    if let Some(amount) = amount_msat {
        builder = builder.amount_milli_satoshis(amount);
    }
    builder
        .build_signed(|hash| Secp256k1::new().sign_ecdsa_recoverable(hash, &secret))
        .unwrap()
        .to_string()
}

fn invoice() -> Invoice {
    Invoice::parse(&signed_invoice(Some(1000), 3600)).unwrap()
}

fn wallet_error(code: &str) -> Reply {
    Err(TransportError::Wallet(WalletServiceError {
        code: code.into(),
        message: code.to_lowercase(),
    }))
}

fn record(direction: &str, state: &str, hash: &str) -> Value {
    json!({
        "type": direction, "state": state, "payment_hash": hash,
        "amount": 1000, "fees_paid": 2, "created_at": 1, "settled_at": 2
    })
}

#[tokio::test]
async fn pay_submits_once_and_reports_the_fee() {
    let wallet = FakeWallet::new();
    let invoice = invoice();
    wallet.reply(wire::LOOKUP_INVOICE, wallet_error("NOT_FOUND"));
    wallet.reply(
        wire::PAY_INVOICE,
        Ok(json!({"preimage": hex::encode(PREIMAGE), "fees_paid": 5})),
    );
    let paid = provider(&wallet).pay(&invoice, 1000, 1100).await.unwrap();
    assert_eq!(paid.status, PaymentStatus::Succeeded);
    assert_eq!(paid.fee_msat, 5);
    assert_eq!(wallet.methods(), [wire::LOOKUP_INVOICE, wire::PAY_INVOICE]);
    assert_eq!(
        wallet.params(wire::PAY_INVOICE),
        json!({"invoice": invoice.bolt11}),
        "an invoice with an amount must not be sent a second amount"
    );
}

#[tokio::test]
async fn amountless_invoice_is_paid_with_the_host_amount() {
    let wallet = FakeWallet::new();
    let invoice = Invoice::parse(&signed_invoice(None, 3600)).unwrap();
    wallet.reply(wire::LOOKUP_INVOICE, wallet_error("NOT_FOUND"));
    wallet.reply(
        wire::PAY_INVOICE,
        Ok(json!({"preimage": hex::encode(PREIMAGE)})),
    );
    provider(&wallet).pay(&invoice, 700, 800).await.unwrap();
    assert_eq!(wallet.params(wire::PAY_INVOICE)["amount"], 700);
}

#[tokio::test]
async fn an_existing_payment_is_returned_instead_of_paying_again() {
    let wallet = FakeWallet::new();
    let invoice = invoice();
    wallet.reply(
        wire::LOOKUP_INVOICE,
        Ok(record("outgoing", "pending", &invoice.payment_hash)),
    );
    let existing = provider(&wallet).pay(&invoice, 1000, 1100).await.unwrap();
    assert_eq!(existing.status, PaymentStatus::Pending);
    assert_eq!(wallet.methods(), [wire::LOOKUP_INVOICE]);
}

#[tokio::test]
async fn own_invoice_is_refused_before_submission() {
    let wallet = FakeWallet::new();
    let invoice = invoice();
    wallet.reply(
        wire::LOOKUP_INVOICE,
        Ok(record("incoming", "pending", &invoice.payment_hash)),
    );
    let error = provider(&wallet)
        .pay(&invoice, 1000, 1100)
        .await
        .unwrap_err();
    assert!(matches!(error, PayError::NotSubmitted(_)), "{error}");
}

#[tokio::test]
async fn failures_before_the_wallet_sees_the_request_are_not_submitted() {
    let invoice = invoice();

    let over_cap = FakeWallet::new();
    let error = provider(&over_cap)
        .pay(&invoice, 1000, 999)
        .await
        .unwrap_err();
    assert!(matches!(error, PayError::NotSubmitted(_)), "{error}");
    assert!(over_cap.methods().is_empty());

    let lookup_down = FakeWallet::new();
    lookup_down.reply(
        wire::LOOKUP_INVOICE,
        Err(TransportError::NoResponse("relay down".into())),
    );
    let error = provider(&lookup_down)
        .pay(&invoice, 1000, 1100)
        .await
        .unwrap_err();
    assert!(matches!(error, PayError::NotSubmitted(_)), "{error}");
    assert_eq!(lookup_down.methods(), [wire::LOOKUP_INVOICE]);

    let unsent = FakeWallet::new();
    unsent.reply(wire::LOOKUP_INVOICE, wallet_error("NOT_FOUND"));
    unsent.reply(
        wire::PAY_INVOICE,
        Err(TransportError::NotSent("no relay".into())),
    );
    let error = provider(&unsent)
        .pay(&invoice, 1000, 1100)
        .await
        .unwrap_err();
    assert!(matches!(error, PayError::NotSubmitted(_)), "{error}");
}

#[tokio::test]
async fn pay_errors_are_classified_by_code() {
    let invoice = invoice();
    for (reply, not_submitted) in [
        (wallet_error("QUOTA_EXCEEDED"), true),
        (wallet_error("INSUFFICIENT_BALANCE"), true),
        (wallet_error("INTERNAL"), false),
        (wallet_error("PAYMENT_FAILED"), false),
        (Err(TransportError::NoResponse("timeout".into())), false),
        (Err(TransportError::Malformed("garbage".into())), false),
        (Ok(json!({"preimage": hex::encode([9u8; 32])})), false),
        (Ok(json!({"preimage": "zz"})), false),
    ] {
        let wallet = FakeWallet::new();
        wallet.reply(wire::LOOKUP_INVOICE, wallet_error("NOT_FOUND"));
        wallet.reply(wire::PAY_INVOICE, reply);
        let error = provider(&wallet)
            .pay(&invoice, 1000, 1100)
            .await
            .unwrap_err();
        assert_eq!(
            matches!(error, PayError::NotSubmitted(_)),
            not_submitted,
            "{error}"
        );
    }
}

#[tokio::test]
async fn created_invoice_must_honor_amount_and_expiry() {
    let wallet = FakeWallet::new();
    let provider = provider(&wallet);
    assert!(provider.create_invoice(None, 300).await.is_err());
    assert!(wallet.methods().is_empty());

    wallet.reply(
        wire::MAKE_INVOICE,
        Ok(json!({"invoice": signed_invoice(Some(1000), 300)})),
    );
    let made = provider.create_invoice(Some(1000), 300).await.unwrap();
    assert_eq!(made.amount_msat, Some(1000));
    assert_eq!(
        wallet.params(wire::MAKE_INVOICE),
        json!({"amount": 1000, "description": "mesh-llm", "expiry": 300})
    );

    wallet.reply(
        wire::MAKE_INVOICE,
        Ok(json!({"invoice": signed_invoice(Some(1000), 86_400)})),
    );
    let error = provider.create_invoice(Some(1000), 300).await.unwrap_err();
    assert!(error.to_string().contains("expiry"), "{error}");

    wallet.reply(
        wire::MAKE_INVOICE,
        Ok(json!({"invoice": signed_invoice(Some(999), 300)})),
    );
    assert!(provider.create_invoice(Some(1000), 300).await.is_err());
}

#[tokio::test]
async fn lookup_maps_not_found_and_rejects_a_different_hash() {
    let wallet = FakeWallet::new();
    let provider = provider(&wallet);
    wallet.reply(wire::LOOKUP_INVOICE, wallet_error("NOT_FOUND"));
    assert_eq!(provider.lookup("aa").await.unwrap(), None);
    wallet.reply(
        wire::LOOKUP_INVOICE,
        Ok(record("outgoing", "settled", "bb")),
    );
    assert!(provider.lookup("aa").await.is_err());
    wallet.reply(
        wire::LOOKUP_INVOICE,
        Ok(record("incoming", "settled", "aa")),
    );
    let found = provider.lookup("aa").await.unwrap().unwrap();
    assert!(found.inbound);
    assert_eq!(found.status, PaymentStatus::Succeeded);
}

#[tokio::test]
async fn settlement_wait_wakes_on_notification_and_rides_out_relay_errors() {
    let wallet = FakeWallet::new();
    wallet.reply(
        wire::LOOKUP_INVOICE,
        Err(TransportError::NoResponse("relay hiccup".into())),
    );
    wallet.reply(
        wire::LOOKUP_INVOICE,
        Ok(record("incoming", "pending", "aa")),
    );
    wallet.reply(
        wire::LOOKUP_INVOICE,
        Ok(record("incoming", "settled", "aa")),
    );
    let provider = provider(&wallet);
    let notifier = Arc::clone(&wallet);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let note: Notification = serde_json::from_value(json!({
                "notification_type": "payment_received",
                "notification": record("incoming", "settled", "aa"),
            }))
            .unwrap();
            let _ = notifier.notifications.send(note);
        }
    });
    let settled = tokio::time::timeout(Duration::from_secs(5), provider.wait_for_payment("aa"))
        .await
        .expect("a notification must wake the wait long before the poll interval")
        .unwrap();
    assert_eq!(settled.status, PaymentStatus::Succeeded);
    assert_eq!(wallet.methods().len(), 3);
}

#[tokio::test]
async fn settlement_wait_stops_on_a_wallet_error() {
    let wallet = FakeWallet::new();
    wallet.reply(wire::LOOKUP_INVOICE, wallet_error("UNAUTHORIZED"));
    let error = provider(&wallet).wait_for_payment("aa").await.unwrap_err();
    assert!(error.to_string().contains("UNAUTHORIZED"), "{error}");
}

#[tokio::test]
async fn nwc321_pay_passes_the_host_fee_headroom_as_max_fee() {
    let wallet = FakeWallet::new();
    let invoice = invoice();
    wallet.reply(wire::LOOKUP_INVOICE, wallet_error("NOT_FOUND"));
    wallet.reply(
        wire::PAY,
        Ok(json!({
            "transaction_id": "tx-1", "state": "settled", "instruction_type": "bolt11",
            "amount": 1000, "fees_paid": 7, "payment_hash": invoice.payment_hash,
            "preimage": hex::encode(PREIMAGE), "created_at": 10, "settled_at": 11
        })),
    );
    let paid = nwc321_provider(&wallet)
        .pay(&invoice, 1000, 1250)
        .await
        .unwrap();
    assert_eq!(paid.status, PaymentStatus::Succeeded);
    assert_eq!(paid.fee_msat, 7);
    assert_eq!(paid.created_at_ms, 10_000);
    assert_eq!(paid.settled_at_ms, Some(11_000));
    assert_eq!(wallet.methods(), [wire::LOOKUP_INVOICE, wire::PAY]);
    assert_eq!(
        wallet.params(wire::PAY),
        json!({"payment": format!("bitcoin:?lightning={}", invoice.bolt11), "max_fee": 250})
    );
}

#[tokio::test]
async fn nwc321_pay_refusals_and_partial_answers_are_classified() {
    let invoice = invoice();
    let settled_without_preimage = json!({"state": "settled", "fees_paid": 1});
    let pending = json!({"transaction_id": "tx", "state": "pending"});
    let failed = json!({"state": "failed", "failure_reason": "no route"});
    let bolt12 = json!({"state": "settled", "instruction_type": "bolt12"});
    for (reply, expected) in [
        (wallet_error("FEE_LIMIT_EXCEEDED"), Err(true)),
        (wallet_error("UNSUPPORTED_PAYMENT_INSTRUCTION"), Err(true)),
        (wallet_error("PAYMENT_FAILED"), Err(false)),
        (Ok(bolt12), Err(false)),
        (Ok(settled_without_preimage), Ok(PaymentStatus::Pending)),
        (Ok(pending), Ok(PaymentStatus::Pending)),
        (Ok(failed), Ok(PaymentStatus::Failed)),
    ] {
        let wallet = FakeWallet::new();
        wallet.reply(wire::LOOKUP_INVOICE, wallet_error("NOT_FOUND"));
        wallet.reply(wire::PAY, reply);
        let result = nwc321_provider(&wallet).pay(&invoice, 1000, 1100).await;
        match (result, expected) {
            (Ok(paid), Ok(status)) => assert_eq!(paid.status, status),
            (Err(error), Err(not_submitted)) => assert_eq!(
                matches!(error, PayError::NotSubmitted(_)),
                not_submitted,
                "{error}"
            ),
            (result, expected) => panic!("{result:?} vs {expected:?}"),
        }
    }
}

#[tokio::test]
async fn amountless_invoice_comes_from_nwc321_receive() {
    let wallet = FakeWallet::new();
    let bolt11 = signed_invoice(None, 3600);
    wallet.reply(
        wire::RECEIVE,
        Ok(json!({"bip321": format!("bitcoin:?lightning={bolt11}&lno=lno1offer")})),
    );
    let made = nwc321_provider(&wallet)
        .create_invoice(None, 86_400)
        .await
        .unwrap();
    assert_eq!(made.amount_msat, None);
    assert_eq!(
        wallet.params(wire::RECEIVE),
        json!({"amount": null, "description": "mesh-llm"})
    );

    // `receive` takes no expiry, so an invoice outliving the request is
    // refused like any other.
    wallet.reply(
        wire::RECEIVE,
        Ok(json!({"bip321": format!("bitcoin:?lightning={bolt11}")})),
    );
    assert!(
        nwc321_provider(&wallet)
            .create_invoice(None, 60)
            .await
            .is_err()
    );

    // Invoices with an amount still use `make_invoice`, which takes an expiry.
    wallet.reply(
        wire::MAKE_INVOICE,
        Ok(json!({"invoice": signed_invoice(Some(1000), 300)})),
    );
    nwc321_provider(&wallet)
        .create_invoice(Some(1000), 300)
        .await
        .unwrap();
    assert_eq!(
        wallet.methods(),
        [wire::RECEIVE, wire::RECEIVE, wire::MAKE_INVOICE]
    );
}
