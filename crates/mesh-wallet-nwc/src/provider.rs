//! `WalletProvider` over a Nostr Wallet Connect service.
//!
//! NWC cannot give two guarantees the host normally relies on:
//!
//! - Core `pay_invoice` takes no fee limit. When the wallet offers NWC-321
//!   `pay`, payments go through it with `max_fee` set to the host's headroom,
//!   but a wallet may ignore `max_fee`, so a payment may still cost more than
//!   that headroom. The fee the wallet reports is recorded as spend, so it
//!   counts against the daily budget either way.
//! - Notifications only report settlement, so there is no claiming signal;
//!   arrival waits end on the terminal status, and `WalletFeatures` says so.
//!
//! Amount-less invoices need NWC-321 `receive`; `make_invoice` requires an
//! amount.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use mesh_llm_wallet::invoice::Invoice;
use mesh_llm_wallet::provider::{Balance, PayError, PaymentStatus, Transaction, WalletProvider};
use serde_json::{Value, json};
use tokio::sync::broadcast;

use crate::transport::{Transport, TransportError};
use crate::wire::{self, Notification, WalletMethods, WalletTransaction};

/// Timing knobs, all overridable from the plugin's args.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// How long to wait for any single wallet response.
    pub request_timeout: Duration,
    /// How long to wait for a `pay_invoice` response. Routing can be slow and
    /// a timeout here leaves the payment uncertain, so it is longer.
    pub pay_timeout: Duration,
    /// How often a settlement wait looks the payment up when no notification
    /// arrives. Notifications are best effort; this is the backstop.
    pub poll_interval: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(20),
            pay_timeout: Duration::from_secs(120),
            poll_interval: Duration::from_secs(3),
        }
    }
}

/// Description attached to every invoice this wallet creates.
const INVOICE_DESCRIPTION: &str = "mesh-llm";
/// Slack between the requested invoice lifetime and the one the wallet
/// produced: BOLT11 timestamps have one-second resolution and the wallet
/// stamps the invoice after it receives the request.
const EXPIRY_TOLERANCE_MS: u64 = 2_000;

pub struct NwcProvider {
    transport: Arc<dyn Transport>,
    timing: Timing,
    methods: WalletMethods,
}

impl NwcProvider {
    pub fn new(transport: Arc<dyn Transport>, timing: Timing, methods: WalletMethods) -> Self {
        Self {
            transport,
            timing,
            methods,
        }
    }

    async fn call(&self, method: &'static str, params: Value) -> Result<Value, TransportError> {
        self.transport
            .request(method, params, self.timing.request_timeout)
            .await
    }

    /// Wait until `settled` holds for the payment, waking on notifications and
    /// otherwise polling. Relay hiccups while waiting are retried: the caller
    /// owns the deadline and cancels by dropping the future.
    async fn wait_until(
        &self,
        payment_hash: &str,
        settled: impl Fn(&Transaction) -> bool,
    ) -> Result<Transaction> {
        // Subscribe before the first lookup so a settlement between the two
        // cannot be missed.
        let mut notifications = self.transport.notifications();
        loop {
            match self.lookup(payment_hash).await {
                Ok(Some(payment)) if settled(&payment) => return Ok(payment),
                Ok(_) => {}
                Err(error) if is_transient(&error) => {
                    tracing::debug!(
                        target: "mesh_wallet_nwc",
                        error = %error,
                        "lookup failed while waiting; retrying"
                    );
                }
                Err(error) => return Err(error),
            }
            tokio::select! {
                () = notified(&mut notifications, payment_hash) => {}
                () = tokio::time::sleep(self.timing.poll_interval) => {}
            }
        }
    }
}

fn is_transient(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<TransportError>(),
        Some(TransportError::NotSent(_) | TransportError::NoResponse(_))
    )
}

/// Resolve when a notification about `payment_hash` arrives. A lagged or
/// closed channel resolves too, since the lookup it triggers is the truth.
async fn notified(notifications: &mut broadcast::Receiver<Notification>, payment_hash: &str) {
    loop {
        match notifications.recv().await {
            Ok(notification) if notification.notification.payment_hash == payment_hash => return,
            Ok(_) => {}
            Err(broadcast::error::RecvError::Lagged(_)) => return,
            Err(broadcast::error::RecvError::Closed) => std::future::pending().await,
        }
    }
}

fn now_secs() -> u64 {
    mesh_llm_wallet::now_ms() / 1000
}

#[async_trait]
impl WalletProvider for NwcProvider {
    async fn balance(&self) -> Result<Balance> {
        let result = self.call(wire::GET_BALANCE, json!({})).await?;
        let balance: wire::Balance = wire::decode(wire::GET_BALANCE, result)?;
        Ok(Balance {
            spendable_msat: balance.balance,
        })
    }

    async fn transactions(&self, limit: usize) -> Result<Vec<Transaction>> {
        let result = self
            .call(wire::LIST_TRANSACTIONS, json!({"limit": limit}))
            .await?;
        let list: wire::TransactionList = wire::decode(wire::LIST_TRANSACTIONS, result)?;
        let now = now_secs();
        Ok(list
            .transactions
            .iter()
            .map(|record| record.to_transaction(now))
            .collect())
    }

    async fn create_invoice(&self, amount_msat: Option<u64>, expiry_secs: u32) -> Result<Invoice> {
        anyhow::ensure!(amount_msat != Some(0), "invoice amount must be positive");
        anyhow::ensure!(expiry_secs > 0, "invoice expiry must be positive");
        let bolt11 = match amount_msat {
            Some(amount_msat) => self.make_invoice(amount_msat, expiry_secs).await?,
            None if self.methods.receive => self.receive_amountless().await?,
            None => bail!(
                "this Nostr Wallet Connect wallet cannot create amount-less invoices (it does \
                 not offer NWC-321 `receive`)"
            ),
        };
        let invoice = Invoice::parse(&bolt11)?;
        // NIP-47 `expiry` is only a request and NWC-321 `receive` takes none.
        // The host relies on short inference invoices expiring at the payee,
        // so an invoice that outlives the requested expiry is refused.
        let latest = mesh_llm_wallet::now_ms()
            .saturating_add(u64::from(expiry_secs) * 1000)
            .saturating_add(EXPIRY_TOLERANCE_MS);
        if invoice.expires_at_ms > latest {
            bail!("wallet ignored the requested invoice expiry of {expiry_secs}s");
        }
        if invoice.amount_msat != amount_msat {
            bail!("wallet created an invoice for a different amount");
        }
        Ok(invoice)
    }

    async fn pay(
        &self,
        invoice: &Invoice,
        amount_msat: u64,
        max_total_msat: u64,
    ) -> Result<Transaction, PayError> {
        invoice
            .validate_payment(amount_msat, mesh_llm_wallet::now_ms())
            .map_err(PayError::NotSubmitted)?;
        let max_fee_msat = max_total_msat.checked_sub(amount_msat).ok_or_else(|| {
            PayError::NotSubmitted(anyhow!("payment amount exceeds the authorized total"))
        })?;
        // The payment hash is the idempotency key: never pay twice.
        if let Some(existing) = self
            .lookup(&invoice.payment_hash)
            .await
            .map_err(PayError::NotSubmitted)?
        {
            if existing.inbound {
                return Err(PayError::NotSubmitted(anyhow!(
                    "cannot pay this wallet's own invoice"
                )));
            }
            return Ok(existing);
        }
        let paid = if self.methods.pay {
            self.pay_with_fee_limit(invoice, amount_msat, max_fee_msat)
                .await?
        } else {
            self.pay_invoice(invoice, amount_msat).await?
        };
        if amount_msat.saturating_add(paid.fee_msat) > max_total_msat {
            tracing::warn!(
                target: "mesh_wallet_nwc",
                fee_msat = paid.fee_msat,
                "payment fee exceeded the host's cap; only the wallet budget bounded it"
            );
        }
        Ok(paid)
    }

    async fn lookup(&self, payment_hash: &str) -> Result<Option<Transaction>> {
        match self
            .call(wire::LOOKUP_INVOICE, json!({"payment_hash": payment_hash}))
            .await
        {
            Ok(result) => {
                let record: WalletTransaction = wire::decode(wire::LOOKUP_INVOICE, result)?;
                if record.payment_hash != payment_hash {
                    bail!("wallet returned a different payment hash");
                }
                Ok(Some(record.to_transaction(now_secs())))
            }
            Err(TransportError::Wallet(error)) if error.is_not_found() => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    async fn wait_for_payment(&self, payment_hash: &str) -> Result<Transaction> {
        self.wait_until(payment_hash, |payment| {
            payment.status != PaymentStatus::Pending
        })
        .await
    }

    // `wait_for_arrival` keeps the default: NWC exposes no claiming state.
}

impl NwcProvider {
    async fn make_invoice(&self, amount_msat: u64, expiry_secs: u32) -> Result<String> {
        let result = self
            .call(
                wire::MAKE_INVOICE,
                json!({
                    "amount": amount_msat,
                    "description": INVOICE_DESCRIPTION,
                    "expiry": expiry_secs,
                }),
            )
            .await?;
        let made: wire::MadeInvoice = wire::decode(wire::MAKE_INVOICE, result)?;
        Ok(made.invoice)
    }

    /// NWC-321 `receive` with no amount: the only way NWC can make an
    /// amount-less invoice.
    async fn receive_amountless(&self) -> Result<String> {
        let result = self
            .call(
                wire::RECEIVE,
                json!({"amount": null, "description": INVOICE_DESCRIPTION}),
            )
            .await?;
        let received: wire::ReceiveResult = wire::decode(wire::RECEIVE, result)?;
        wire::bolt11_from_uri(&received.bip321)
    }

    /// Core NIP-47 `pay_invoice`. It has no fee limit; only the connection's
    /// budget bounds the fee.
    async fn pay_invoice(
        &self,
        invoice: &Invoice,
        amount_msat: u64,
    ) -> Result<Transaction, PayError> {
        let mut params = json!({"invoice": invoice.bolt11});
        if invoice.amount_msat.is_none() {
            params["amount"] = json!(amount_msat);
        }
        let result = self
            .transport
            .request(wire::PAY_INVOICE, params, self.timing.pay_timeout)
            .await
            .map_err(submission_error)?;
        let paid: wire::PaidInvoice = wire::decode(wire::PAY_INVOICE, result)
            .map_err(|error| PayError::Uncertain(error.context("payment outcome uncertain")))?;
        verify_preimage(invoice, &paid.preimage)?;
        let now_ms = mesh_llm_wallet::now_ms();
        Ok(outgoing(
            invoice,
            amount_msat,
            paid.fees_paid.unwrap_or(0),
            PaymentStatus::Succeeded,
            now_ms,
            Some(now_ms),
        ))
    }

    /// NWC-321 `pay` with `max_fee`, so a wallet that implements fee limits
    /// refuses a route that would exceed the host's cap (`FEE_LIMIT_EXCEEDED`,
    /// nothing sent). Wallets may ignore `max_fee`, so this narrows the risk
    /// the connection budget covers rather than removing it.
    async fn pay_with_fee_limit(
        &self,
        invoice: &Invoice,
        amount_msat: u64,
        max_fee_msat: u64,
    ) -> Result<Transaction, PayError> {
        let mut params = json!({
            "payment": wire::bolt11_payment_uri(&invoice.bolt11),
            "max_fee": max_fee_msat,
        });
        if invoice.amount_msat.is_none() {
            params["amount"] = json!(amount_msat);
        }
        let result = self
            .transport
            .request(wire::PAY, params, self.timing.pay_timeout)
            .await
            .map_err(submission_error)?;
        let paid: wire::PayResult = wire::decode(wire::PAY, result)
            .map_err(|error| PayError::Uncertain(error.context("payment outcome uncertain")))?;
        if paid
            .instruction_type
            .as_deref()
            .is_some_and(|kind| kind != "bolt11")
            || paid
                .payment_hash
                .as_deref()
                .is_some_and(|hash| hash != invoice.payment_hash)
        {
            return Err(PayError::Uncertain(anyhow!(
                "wallet reports paying something other than this invoice"
            )));
        }
        if paid.fees_paid.is_none() {
            tracing::warn!(
                target: "mesh_wallet_nwc",
                "wallet did not report fees_paid; it may not have enforced max_fee"
            );
        }
        let now_ms = mesh_llm_wallet::now_ms();
        let created_at_ms = paid.created_at.map_or(now_ms, |at| at.saturating_mul(1000));
        let status = match (paid.state.as_deref(), paid.preimage.as_deref()) {
            (Some("settled"), Some(preimage)) => {
                verify_preimage(invoice, preimage)?;
                PaymentStatus::Succeeded
            }
            (Some("failed"), _) => PaymentStatus::Failed,
            // Pending, unknown, or settled without a preimage to check: the
            // host confirms by payment hash before recording anything.
            _ => PaymentStatus::Pending,
        };
        let settled_at_ms = (status != PaymentStatus::Pending)
            .then(|| paid.settled_at.map_or(now_ms, |at| at.saturating_mul(1000)));
        let mut transaction = outgoing(
            invoice,
            amount_msat,
            paid.fees_paid.unwrap_or(0),
            status,
            created_at_ms,
            settled_at_ms,
        );
        transaction.status_msg = paid.failure_reason.or(paid.state);
        Ok(transaction)
    }
}

/// Classify a failed payment request. Only a request that never reached a
/// relay, or a refusal the wallet makes before routing, proves nothing moved.
fn submission_error(error: TransportError) -> PayError {
    match error {
        TransportError::NotSent(reason) => {
            PayError::NotSubmitted(anyhow!("request not sent: {reason}"))
        }
        TransportError::Wallet(error) if error.proves_not_submitted() => {
            PayError::NotSubmitted(anyhow!("wallet refused payment: {error}"))
        }
        other => PayError::Uncertain(anyhow!(
            "payment outcome uncertain; reconcile by payment hash: {other}"
        )),
    }
}

fn verify_preimage(invoice: &Invoice, preimage: &str) -> Result<(), PayError> {
    let preimage: [u8; 32] = hex::decode(preimage)
        .ok()
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| PayError::Uncertain(anyhow!("wallet returned a malformed preimage")))?;
    if !invoice.verifies_preimage(&preimage) {
        return Err(PayError::Uncertain(anyhow!(
            "wallet returned a preimage that does not match the invoice"
        )));
    }
    Ok(())
}

fn outgoing(
    invoice: &Invoice,
    amount_msat: u64,
    fee_msat: u64,
    status: PaymentStatus,
    created_at_ms: u64,
    settled_at_ms: Option<u64>,
) -> Transaction {
    Transaction {
        id: invoice.payment_hash.clone(),
        payment_hash: Some(invoice.payment_hash.clone()),
        inbound: false,
        amount_msat,
        fee_msat,
        status,
        claiming: false,
        status_msg: None,
        created_at_ms,
        settled_at_ms,
    }
}

#[cfg(test)]
#[path = "provider_tests.rs"]
mod tests;
