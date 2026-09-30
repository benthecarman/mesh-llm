//! Nostr Wallet Connect (NIP-47) as a built-in mesh-llm wallet plugin.
//!
//! Like `wallet-lexe`, this runs as a separate process the host launches as
//! `mesh-llm --plugin wallet-nwc` and serves the `wallet.v1` capability. It
//! holds no seed: the wallet lives in whatever NWC service the operator
//! connects, and the plugin only needs the connection URI.
//!
//! Configure it with a `[[plugin]]` stanza; it does not run otherwise. Its
//! `args` are parsed by [`NwcArgs`]:
//!
//! ```toml
//! [[plugin]]
//! name = "wallet-nwc"
//! args = ["--uri-file", "/home/me/.mesh-llm/nwc-uri"]
//! ```
#![forbid(unsafe_code)]

mod provider;
mod transport;
mod wire;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use clap::Parser;
use mesh_llm_plugin::PluginRuntime;
use mesh_llm_wallet::backend::{OpenedWallet, WalletBackend};
use mesh_llm_wallet::contract::{WalletFeatures, WalletIdentity};
use mesh_llm_wallet::plugin_server::wallet_plugin;
use nostr::nips::nip47::NostrWalletConnectUri;

use crate::provider::{NwcProvider, Timing};
use crate::transport::{RelayTransport, Transport, TransportError};
use crate::wire::WalletMethods;

/// Plugin name the host launches this implementation under.
pub const PLUGIN_NAME: &str = "wallet-nwc";
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The `args` of a `wallet-nwc` `[[plugin]]` stanza.
#[derive(Debug, Parser)]
#[command(name = PLUGIN_NAME, no_binary_name = true)]
struct NwcArgs {
    /// Absolute path to a file holding the `nostr+walletconnect://` URI. The
    /// URI carries spending authority, so it is never taken inline; keep the
    /// file at mode 0600.
    #[arg(long, value_name = "PATH")]
    uri_file: PathBuf,
    /// Seconds to wait for a wallet response.
    #[arg(long, value_name = "SECS", value_parser = clap::value_parser!(u64).range(1..))]
    request_timeout_secs: Option<u64>,
    /// Seconds to wait for a payment response before treating it as uncertain.
    #[arg(long, value_name = "SECS", value_parser = clap::value_parser!(u64).range(1..))]
    pay_timeout_secs: Option<u64>,
    /// Milliseconds between payment lookups when no notification arrives.
    #[arg(long, value_name = "MS", value_parser = clap::value_parser!(u64).range(1..))]
    poll_interval_ms: Option<u64>,
}

impl NwcArgs {
    fn parse(args: &[String]) -> Result<Self> {
        let parsed = Self::try_parse_from(args)
            .map_err(|error| anyhow::anyhow!("invalid wallet-nwc args: {error}"))?;
        if !parsed.uri_file.is_absolute() {
            bail!("wallet-nwc `--uri-file` must be an absolute path");
        }
        Ok(parsed)
    }

    fn timing(&self) -> Timing {
        let defaults = Timing::default();
        Timing {
            request_timeout: self
                .request_timeout_secs
                .map_or(defaults.request_timeout, Duration::from_secs),
            pay_timeout: self
                .pay_timeout_secs
                .map_or(defaults.pay_timeout, Duration::from_secs),
            poll_interval: self
                .poll_interval_ms
                .map_or(defaults.poll_interval, Duration::from_millis),
        }
    }
}

/// Read the connection URI. Errors never include the file's contents: the
/// URI's secret is the wallet's spending credential.
fn read_uri(path: &Path) -> Result<NostrWalletConnectUri> {
    warn_if_readable_by_others(path);
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read NWC connection URI from {}", path.display()))?;
    NostrWalletConnectUri::parse(raw.trim()).map_err(|_| {
        anyhow::anyhow!(
            "{} does not hold a valid nostr+walletconnect:// URI",
            path.display()
        )
    })
}

#[cfg(unix)]
fn warn_if_readable_by_others(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(metadata) = std::fs::metadata(path)
        && metadata.permissions().mode() & 0o077 != 0
    {
        tracing::warn!(
            target: "mesh_wallet_nwc",
            path = %path.display(),
            "the NWC URI file is readable by other users; restrict it to mode 0600"
        );
    }
}

#[cfg(not(unix))]
fn warn_if_readable_by_others(_path: &Path) {}

/// What the wallet says about itself: the `get_info` result, or, when the
/// connection may not call `get_info`, the methods the wallet's info event
/// lists with no network or node key.
fn wallet_info(
    get_info: Result<serde_json::Value, TransportError>,
    advertised_methods: &[String],
) -> Result<wire::Info> {
    match get_info {
        Ok(result) => wire::decode(wire::GET_INFO, result),
        Err(TransportError::Wallet(error))
            if matches!(error.code.as_str(), "RESTRICTED" | "NOT_IMPLEMENTED") =>
        {
            tracing::warn!(
                target: "mesh_wallet_nwc",
                code = %error.code,
                "the NWC connection may not call get_info; assuming mainnet and identifying \
                 the wallet by the connection's key"
            );
            Ok(wire::Info {
                methods: advertised_methods.to_vec(),
                ..wire::Info::default()
            })
        }
        Err(error) => Err(anyhow::Error::new(error).context("NWC get_info failed")),
    }
}

/// Refuse a connection that cannot do what paid inference needs, naming the
/// permissions to grant rather than failing at the first payment.
fn ensure_required_methods(info: &wire::Info) -> Result<()> {
    // A wallet that does not list methods is not refused; its first
    // unsupported request will say so.
    if info.methods.is_empty() {
        return Ok(());
    }
    let allows = |method: &str| info.methods.iter().any(|allowed| allowed == method);
    let mut missing: Vec<&str> = wire::REQUIRED_METHODS
        .iter()
        .copied()
        .filter(|method| !allows(method))
        .collect();
    if !allows(wire::PAY_INVOICE) && !allows(wire::PAY) {
        missing.push("pay_invoice (or NWC-321 pay)");
    }
    if !missing.is_empty() {
        bail!(
            "the NWC connection does not allow {}; grant these permissions in the wallet",
            missing.join(", ")
        );
    }
    Ok(())
}

/// NIP-47 names networks the way [`WalletIdentity::network`] does. Most
/// wallets report it; one that does not is taken to be on mainnet.
fn network_of(info: &wire::Info) -> String {
    info.network
        .as_deref()
        .map(str::trim)
        .filter(|network| !network.is_empty())
        .unwrap_or("mainnet")
        .to_owned()
}

/// The node key when the wallet reports one, so rotating the connection to
/// the same wallet keeps the host's pin; otherwise the connection's key.
fn identity_of(info: &wire::Info, uri: &NostrWalletConnectUri) -> WalletIdentity {
    let wallet_id = info
        .pubkey
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map_or_else(|| uri.public_key.to_hex(), str::to_owned);
    WalletIdentity {
        wallet_id,
        provider: "nwc".into(),
        network: network_of(info),
    }
}

fn features(methods: WalletMethods) -> WalletFeatures {
    WalletFeatures {
        // `make_invoice` requires an amount; NWC-321 `receive` does not.
        amountless_invoices: methods.receive,
    }
}

struct NwcBackend {
    args: NwcArgs,
}

#[async_trait]
impl WalletBackend for NwcBackend {
    fn provider_name(&self) -> &'static str {
        "nwc"
    }

    async fn open(&self, _directory: &Path) -> Result<OpenedWallet> {
        // The wallet's state lives in the NWC service; nothing is stored in
        // the host-supplied directory.
        let timing = self.args.timing();
        let uri = read_uri(&self.args.uri_file)?;
        let transport = RelayTransport::connect(&uri, timing.request_timeout).await?;
        let get_info = transport
            .request(
                wire::GET_INFO,
                serde_json::json!({}),
                timing.request_timeout,
            )
            .await;
        let info = wallet_info(get_info, transport.advertised_methods())?;
        ensure_required_methods(&info)?;
        let methods = WalletMethods::from_info(&info);
        Ok(OpenedWallet {
            identity: identity_of(&info, &uri),
            provider: Arc::new(NwcProvider::new(Arc::new(transport), timing, methods)),
            created: false,
            features: features(methods),
        })
    }
}

/// Serve the `wallet.v1` capability on the host-supplied plugin endpoint until
/// the host closes the connection. `name` is the plugin name the host launched
/// this process under and `args` its `[[plugin]]` stanza's `args`. Logging is
/// owned by the launching binary.
pub async fn run_plugin(name: String, args: Vec<String>) -> Result<()> {
    let backend = NwcBackend {
        args: NwcArgs::parse(&args)?,
    };
    PluginRuntime::run(wallet_plugin(name, VERSION, backend)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    const URI: &str = "nostr+walletconnect://b889ff5b1513b641e2a139f661a661364979c5beee91842f8f0ef42ab558e9d4?relay=wss%3A%2F%2Frelay.example.com&secret=71a8c14c1407c113601079c4302dab36460f0ccd0ad506f1f2dc73b5100e4f3c";

    fn info(value: serde_json::Value) -> wire::Info {
        serde_json::from_value(value).unwrap()
    }

    fn args(args: &[&str]) -> Result<NwcArgs> {
        NwcArgs::parse(&args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn args_require_an_absolute_uri_file_and_reject_unknown_flags() {
        let missing = args(&[]).unwrap_err().to_string();
        assert!(missing.contains("--uri-file"), "{missing}");
        assert!(args(&["--uri-file", "nwc-uri"]).is_err());
        assert!(
            args(&["--uri", URI]).is_err(),
            "an inline URI must not be accepted"
        );
        assert!(args(&["--uri-file", "/x", "--poll-interval-ms", "0"]).is_err());
        args(&["--uri-file", "/home/me/nwc-uri"]).unwrap();
    }

    #[test]
    fn timing_defaults_can_be_overridden() {
        let timing = args(&["--uri-file", "/x", "--poll-interval-ms", "500"])
            .unwrap()
            .timing();
        assert_eq!(timing.poll_interval, Duration::from_millis(500));
        assert_eq!(timing.request_timeout, Timing::default().request_timeout);
    }

    #[test]
    fn uri_errors_do_not_leak_the_secret() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("uri");
        let broken = URI.replace("relay=", "rel=");
        std::fs::write(&path, &broken).unwrap();
        let error = format!("{:#}", read_uri(&path).unwrap_err());
        assert!(!error.contains("71a8c14c"), "{error}");

        std::fs::write(&path, format!("{URI}\n")).unwrap();
        let uri = read_uri(&path).unwrap();
        assert_eq!(uri.relays.len(), 1);
    }

    #[test]
    fn a_connection_without_get_info_falls_back_to_the_advertised_methods() {
        let refused = |code: &str| {
            Err(TransportError::Wallet(wire::WalletServiceError {
                code: code.into(),
                message: String::new(),
            }))
        };
        let advertised = ["pay".to_owned(), "make_invoice".to_owned()];
        for code in ["RESTRICTED", "NOT_IMPLEMENTED"] {
            let info = wallet_info(refused(code), &advertised).unwrap();
            assert_eq!(info.methods, advertised);
            assert_eq!(network_of(&info), "mainnet");
            assert!(info.pubkey.is_none());
        }

        let reported = wallet_info(
            Ok(serde_json::json!({"network": "signet", "methods": ["pay_invoice"]})),
            &advertised,
        )
        .unwrap();
        assert_eq!(reported.methods, ["pay_invoice"]);
        assert_eq!(reported.network.as_deref(), Some("signet"));

        assert!(wallet_info(refused("UNAUTHORIZED"), &advertised).is_err());
        let silent = wallet_info(
            Err(TransportError::NoResponse("timeout".into())),
            &advertised,
        );
        assert!(silent.is_err(), "only a refusal may fall back");
    }

    #[test]
    fn missing_permissions_are_named() {
        let error = ensure_required_methods(&info(serde_json::json!({
            "methods": ["get_balance", "make_invoice"]
        })))
        .unwrap_err()
        .to_string();
        assert!(error.contains("pay_invoice"), "{error}");
        assert!(error.contains("lookup_invoice"), "{error}");
        ensure_required_methods(&info(serde_json::json!({}))).unwrap();
    }

    #[test]
    fn identity_prefers_the_node_key_and_network_defaults_to_mainnet() {
        let uri = NostrWalletConnectUri::parse(URI).unwrap();
        let bare = identity_of(&info(serde_json::json!({})), &uri);
        assert_eq!(bare.network, "mainnet");
        assert_eq!(bare.wallet_id, uri.public_key.to_hex());
        assert_eq!(bare.provider, "nwc");

        let node = identity_of(
            &info(serde_json::json!({"pubkey": "02abc", "network": " signet "})),
            &uri,
        );
        assert_eq!(node.wallet_id, "02abc");
        assert_eq!(node.network, "signet");
    }

    #[test]
    fn a_wallet_offering_only_nwc321_pay_is_accepted() {
        ensure_required_methods(&info(serde_json::json!({
            "methods": ["pay", "make_invoice", "lookup_invoice", "get_balance"]
        })))
        .unwrap();
        let error = ensure_required_methods(&info(serde_json::json!({
            "methods": ["make_invoice", "lookup_invoice", "get_balance"]
        })))
        .unwrap_err()
        .to_string();
        assert!(error.contains("pay_invoice (or NWC-321 pay)"), "{error}");
    }

    #[test]
    fn features_admit_what_nwc_cannot_do() {
        let features = features(WalletMethods::default());
        assert!(!features.amountless_invoices);
        let with_nwc321 = features_for_nwc321();
        assert!(with_nwc321.amountless_invoices);
    }

    fn features_for_nwc321() -> WalletFeatures {
        features(WalletMethods {
            pay: true,
            receive: true,
        })
    }
}
