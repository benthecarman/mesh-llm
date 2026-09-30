//! NIP-47 payloads as this plugin reads them.
//!
//! These are deliberately more lenient than the `nostr` crate's typed NIP-47
//! structs: wallets in the wild send error codes and transaction states that
//! postdate any fixed enum (`BAD_REQUEST`, `FEE_LIMIT_EXCEEDED`, `accepted`),
//! and an unknown code must still be classified rather than fail decoding.

use anyhow::{Result, bail};
use mesh_llm_wallet::provider::{PaymentStatus, Transaction};
use serde::Deserialize;
use serde_json::Value;

pub const PAY_INVOICE: &str = "pay_invoice";
pub const MAKE_INVOICE: &str = "make_invoice";
pub const LOOKUP_INVOICE: &str = "lookup_invoice";
pub const LIST_TRANSACTIONS: &str = "list_transactions";
pub const GET_BALANCE: &str = "get_balance";
pub const GET_INFO: &str = "get_info";
/// NWC-321: pay a BIP-321 URI, with a routing fee limit.
pub const PAY: &str = "pay";
/// NWC-321: create a BIP-321 URI, amount optional.
pub const RECEIVE: &str = "receive";

/// Methods the wallet must allow for this plugin to back paid inference, in
/// addition to one of [`PAY_INVOICE`] or [`PAY`].
pub const REQUIRED_METHODS: &[&str] = &[MAKE_INVOICE, LOOKUP_INVOICE, GET_BALANCE];

/// Optional methods this plugin uses when the wallet offers them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalletMethods {
    /// NWC-321 `pay`, which takes a `max_fee`.
    pub pay: bool,
    /// NWC-321 `receive`, which can create an amount-less invoice.
    pub receive: bool,
}

impl WalletMethods {
    pub fn from_info(info: &Info) -> Self {
        let offers = |method: &str| info.methods.iter().any(|offered| offered == method);
        Self {
            pay: offers(PAY),
            receive: offers(RECEIVE),
        }
    }
}

/// A kind 23195 response.
#[derive(Clone, Debug, Deserialize)]
pub struct Response {
    #[serde(default)]
    pub result_type: Option<String>,
    #[serde(default)]
    pub error: Option<WalletServiceError>,
    #[serde(default)]
    pub result: Option<Value>,
}

/// A NIP-47 error object. The code is kept as text so codes this plugin does
/// not know are still reported and classified.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct WalletServiceError {
    pub code: String,
    #[serde(default)]
    pub message: String,
}

impl WalletServiceError {
    /// Whether this error guarantees the wallet did not send a payment.
    ///
    /// Only codes that describe a refusal before any routing attempt qualify.
    /// `PAYMENT_FAILED`, `INTERNAL`, `OTHER` and anything unknown may follow a
    /// submitted payment (some wallets report "already paid" as `INTERNAL`),
    /// so they stay uncertain and the host reconciles by payment hash.
    pub fn proves_not_submitted(&self) -> bool {
        matches!(
            self.code.as_str(),
            "RATE_LIMITED"
                | "NOT_IMPLEMENTED"
                | "INSUFFICIENT_BALANCE"
                | "QUOTA_EXCEEDED"
                | "RESTRICTED"
                | "UNAUTHORIZED"
                | "UNSUPPORTED_ENCRYPTION"
                | "BAD_REQUEST"
                | "FEE_LIMIT_EXCEEDED"
                | "UNSUPPORTED_PAYMENT_INSTRUCTION"
                | "UNSUPPORTED_NETWORK"
        )
    }

    pub fn is_not_found(&self) -> bool {
        self.code == "NOT_FOUND"
    }
}

impl std::fmt::Display for WalletServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

/// `get_info` result. Every field is optional in practice.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Info {
    #[serde(default)]
    pub pubkey: Option<String>,
    #[serde(default)]
    pub network: Option<String>,
    #[serde(default)]
    pub methods: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Balance {
    /// Millisatoshis.
    pub balance: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MadeInvoice {
    pub invoice: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PaidInvoice {
    pub preimage: String,
    #[serde(default)]
    pub fees_paid: Option<u64>,
}

/// NWC-321 `pay` result. Its `transaction_id` is not kept: this plugin
/// identifies every payment by its hash, which `lookup_invoice` also uses.
#[derive(Clone, Debug, Deserialize)]
pub struct PayResult {
    /// `pending`, `settled` or `failed`.
    #[serde(default)]
    pub state: Option<String>,
    /// `bolt11` or `bolt12`.
    #[serde(default)]
    pub instruction_type: Option<String>,
    #[serde(default)]
    pub fees_paid: Option<u64>,
    #[serde(default)]
    pub payment_hash: Option<String>,
    #[serde(default)]
    pub preimage: Option<String>,
    #[serde(default)]
    pub failure_reason: Option<String>,
    /// Seconds.
    #[serde(default)]
    pub created_at: Option<u64>,
    /// Seconds.
    #[serde(default)]
    pub settled_at: Option<u64>,
}

/// NWC-321 `receive` result.
#[derive(Clone, Debug, Deserialize)]
pub struct ReceiveResult {
    pub bip321: String,
}

/// A BIP-321 URI carrying exactly one BOLT11 instruction and no on-chain
/// address, so a wallet can only pay that invoice.
pub fn bolt11_payment_uri(bolt11: &str) -> String {
    format!("bitcoin:?lightning={bolt11}")
}

/// The BOLT11 invoice in a BIP-321 URI. The host settles BOLT11 only, so a
/// URI without one (for example a BOLT12 offer alone) is refused.
pub fn bolt11_from_uri(uri: &str) -> Result<String> {
    let (_, rest) = uri
        .split_once(':')
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bitcoin"))
        .ok_or_else(|| anyhow::anyhow!("not a bitcoin: URI"))?;
    let query = rest.split_once('?').map_or("", |(_, query)| query);
    for pair in query.split('&') {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        if key.eq_ignore_ascii_case("lightning") && !value.is_empty() {
            return percent_decode(value);
        }
    }
    bail!("the wallet's payment URI has no BOLT11 invoice")
}

fn percent_decode(value: &str) -> Result<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let hex = value
                .get(index + 1..index + 3)
                .ok_or_else(|| anyhow::anyhow!("truncated percent escape"))?;
            decoded.push(u8::from_str_radix(hex, 16)?);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    Ok(String::from_utf8(decoded)?)
}

#[derive(Clone, Debug, Deserialize)]
pub struct TransactionList {
    #[serde(default)]
    pub transactions: Vec<WalletTransaction>,
}

/// A `lookup_invoice` / `list_transactions` record, also the body of a
/// `payment_received` / `payment_sent` notification.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct WalletTransaction {
    #[serde(rename = "type", default)]
    pub direction: Option<String>,
    #[serde(default)]
    pub state: Option<String>,
    pub payment_hash: String,
    #[serde(default)]
    pub preimage: Option<String>,
    /// Millisatoshis.
    #[serde(default)]
    pub amount: u64,
    #[serde(default)]
    pub fees_paid: Option<u64>,
    /// Seconds.
    #[serde(default)]
    pub created_at: u64,
    #[serde(default)]
    pub settled_at: Option<u64>,
    #[serde(default)]
    pub expires_at: Option<u64>,
}

/// How long after an unpaid incoming invoice's expiry a wallet that does not
/// report `state` is trusted to have recorded a payment that arrived just in
/// time.
const EXPIRY_GRACE_SECS: u64 = 60;

impl WalletTransaction {
    pub fn is_inbound(&self) -> bool {
        self.direction.as_deref() == Some("incoming")
    }

    /// Normalize onto the provider-neutral transaction.
    ///
    /// Wallets that predate the `state` field are read conservatively: an
    /// outgoing payment without a preimage stays pending (invoice expiry does
    /// not end an in-flight HTLC), and an incoming invoice only counts as
    /// failed well after it could have been paid.
    pub fn to_transaction(&self, now_secs: u64) -> Transaction {
        let inbound = self.is_inbound();
        let status = match self.state.as_deref() {
            Some("settled") => PaymentStatus::Succeeded,
            Some("failed" | "expired" | "canceled" | "cancelled") => PaymentStatus::Failed,
            Some(_) => PaymentStatus::Pending,
            None => self.inferred_status(inbound, now_secs),
        };
        Transaction {
            id: self.payment_hash.clone(),
            payment_hash: Some(self.payment_hash.clone()),
            inbound,
            amount_msat: self.amount,
            fee_msat: self.fees_paid.unwrap_or(0),
            status,
            // Only hold invoices expose an accepted-but-unsettled state, and
            // this plugin does not create them.
            claiming: false,
            status_msg: self.state.clone(),
            created_at_ms: self.created_at.saturating_mul(1000),
            settled_at_ms: self.settled_at.map(|at| at.saturating_mul(1000)),
        }
    }

    fn inferred_status(&self, inbound: bool, now_secs: u64) -> PaymentStatus {
        if self.settled_at.is_some() || (!inbound && self.preimage.is_some()) {
            return PaymentStatus::Succeeded;
        }
        let long_expired = self
            .expires_at
            .is_some_and(|at| at.saturating_add(EXPIRY_GRACE_SECS) < now_secs);
        if inbound && long_expired {
            PaymentStatus::Failed
        } else {
            PaymentStatus::Pending
        }
    }
}

/// A kind 23196/23197 notification. Only the payment it names matters: it
/// wakes a settlement wait, which then asks the wallet for the truth.
#[derive(Clone, Debug, Deserialize)]
pub struct Notification {
    pub notification: WalletTransaction,
}

/// Decode a response into its result, or the wallet's error.
pub fn into_result(response: Response, method: &str) -> Result<Value, ResponseError> {
    if let Some(error) = response.error {
        return Err(ResponseError::Wallet(error));
    }
    if let Some(result_type) = &response.result_type
        && result_type != method
    {
        return Err(ResponseError::Malformed(format!(
            "expected a {method} response, got {result_type}"
        )));
    }
    response
        .result
        .ok_or_else(|| ResponseError::Malformed(format!("{method} response has no result")))
}

#[derive(Debug)]
pub enum ResponseError {
    Wallet(WalletServiceError),
    Malformed(String),
}

pub fn decode<T: serde::de::DeserializeOwned>(method: &str, value: Value) -> Result<T> {
    match serde_json::from_value(value) {
        Ok(decoded) => Ok(decoded),
        Err(error) => bail!("malformed {method} result: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(value: Value) -> WalletTransaction {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn only_pre_routing_refusals_prove_nothing_was_sent() {
        let error = |code: &str| WalletServiceError {
            code: code.into(),
            message: String::new(),
        };
        for code in [
            "RATE_LIMITED",
            "QUOTA_EXCEEDED",
            "INSUFFICIENT_BALANCE",
            "RESTRICTED",
            "FEE_LIMIT_EXCEEDED",
            "UNSUPPORTED_PAYMENT_INSTRUCTION",
            "UNSUPPORTED_NETWORK",
        ] {
            assert!(error(code).proves_not_submitted(), "{code}");
        }
        for code in ["PAYMENT_FAILED", "INTERNAL", "OTHER", "SOMETHING_NEW"] {
            assert!(!error(code).proves_not_submitted(), "{code}");
        }
    }

    #[test]
    fn explicit_states_map_directly() {
        let settled = record(json!({
            "type": "incoming", "state": "settled", "payment_hash": "h",
            "amount": 5000, "created_at": 10, "settled_at": 12
        }))
        .to_transaction(100);
        assert_eq!(settled.status, PaymentStatus::Succeeded);
        assert!(settled.inbound);
        assert_eq!(settled.created_at_ms, 10_000);
        assert_eq!(settled.settled_at_ms, Some(12_000));

        let expired = record(json!({
            "type": "incoming", "state": "expired", "payment_hash": "h", "amount": 1
        }))
        .to_transaction(0);
        assert_eq!(expired.status, PaymentStatus::Failed);

        let accepted = record(json!({
            "type": "incoming", "state": "accepted", "payment_hash": "h", "amount": 1
        }))
        .to_transaction(0);
        assert_eq!(accepted.status, PaymentStatus::Pending);
        assert!(!accepted.claiming);
    }

    #[test]
    fn stateless_outgoing_payment_without_preimage_stays_pending_after_expiry() {
        let outgoing = record(json!({
            "type": "outgoing", "payment_hash": "h", "amount": 1, "expires_at": 10
        }));
        assert_eq!(
            outgoing.to_transaction(10_000).status,
            PaymentStatus::Pending
        );
        let paid = record(json!({
            "type": "outgoing", "payment_hash": "h", "amount": 1, "preimage": "00", "fees_paid": 3
        }))
        .to_transaction(0);
        assert_eq!(paid.status, PaymentStatus::Succeeded);
        assert_eq!(paid.fee_msat, 3);
    }

    #[test]
    fn stateless_incoming_invoice_fails_only_after_a_grace_period() {
        let invoice = record(json!({
            "type": "incoming", "payment_hash": "h", "amount": 1, "expires_at": 100
        }));
        assert_eq!(invoice.to_transaction(120).status, PaymentStatus::Pending);
        assert_eq!(invoice.to_transaction(200).status, PaymentStatus::Failed);
    }

    #[test]
    fn bip321_uris_round_trip_a_bolt11_invoice() {
        let uri = bolt11_payment_uri("lnbc10n1abc");
        assert_eq!(uri, "bitcoin:?lightning=lnbc10n1abc");
        assert_eq!(bolt11_from_uri(&uri).unwrap(), "lnbc10n1abc");
        assert_eq!(
            bolt11_from_uri("BITCOIN:bc1qxyz?lno=lno1offer&LIGHTNING=lnbc1%70q").unwrap(),
            "lnbc1pq"
        );
        assert!(bolt11_from_uri("bitcoin:?lno=lno1offer").is_err());
        assert!(bolt11_from_uri("lightning:lnbc1").is_err());
        assert!(bolt11_from_uri("bitcoin:?lightning=lnbc%7").is_err());
    }

    #[test]
    fn methods_are_read_from_get_info() {
        let info: Info = serde_json::from_value(json!({
            "methods": ["pay", "make_invoice", "receive"]
        }))
        .unwrap();
        assert_eq!(
            WalletMethods::from_info(&info),
            WalletMethods {
                pay: true,
                receive: true
            }
        );
        assert_eq!(
            WalletMethods::from_info(&Info::default()),
            WalletMethods::default()
        );
    }

    #[test]
    fn unknown_error_codes_still_decode() {
        let response: Response = serde_json::from_value(json!({
            "result_type": "pay_invoice",
            "error": {"code": "SOMETHING_NEW", "message": "nope"}
        }))
        .unwrap();
        match into_result(response, PAY_INVOICE) {
            Err(ResponseError::Wallet(error)) => assert_eq!(error.code, "SOMETHING_NEW"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn mismatched_result_type_is_malformed() {
        let response: Response = serde_json::from_value(json!({
            "result_type": "get_balance", "result": {"balance": 1}
        }))
        .unwrap();
        assert!(matches!(
            into_result(response, PAY_INVOICE),
            Err(ResponseError::Malformed(_))
        ));
    }
}
