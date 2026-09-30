# mesh-wallet-nwc

Nostr Wallet Connect (NIP-47) as a built-in mesh-llm wallet plugin. Serves the
`wallet.v1` capability (see `crates/mesh-llm-wallet`) from the mesh-llm
executable as `mesh-llm --plugin wallet-nwc`, like `wallet-lexe`.

It holds no seed. The wallet lives in whatever NWC service you connect (Alby
Hub, Coinos, LNbits, ...); this plugin only needs the connection URI.

Like `wallet-lexe`, it is compiled out of default builds. Opt in with
`MESH_LLM_WALLET_NWC=1 just build` or `MESH_LLM_WALLET_NWC=1 just release-build`.
When compiled in, it still does not run until configured:

```toml
[[plugin]]
name = "wallet-nwc"
# The file holds the nostr+walletconnect://... URI.
args = ["--uri-file", "/home/me/.mesh-llm/nwc-uri"]
```

It reads these arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--uri-file <PATH>` | required | Absolute path to the file holding the connection URI |
| `--request-timeout-secs <SECS>` | 20 | How long to wait for a wallet response |
| `--pay-timeout-secs <SECS>` | 120 | How long to wait for a payment response before treating it as uncertain |
| `--poll-interval-ms <MS>` | 3000 | Time between payment lookups when no notification arrives |

A change takes effect when mesh-llm restarts.

Create the connection in your wallet with the `pay_invoice` (or NWC-321 `pay`),
`make_invoice`, `lookup_invoice` and `get_balance` permissions
(`list_transactions` for `mesh-llm wallet get-transactions`, NWC-321 `receive`
for amount-less invoices), and give it a spending budget. Keep the
URI file at mode 0600: the URI is the connection's spending credential, which
is why it is never accepted inline.

Also grant `get_info` where the wallet lets you. It reports the connection's
permissions, the wallet's network and its node key. Without it the plugin
still opens: it reads the methods from the wallet's info event, assumes
mainnet, and identifies the wallet by the connection's key. The plugin does
not check invoice networks itself; it relies on the wallet to refuse an
invoice for another network, so point it only at a mainnet wallet.

What NWC cannot guarantee, and how the plugin reports it:

- **Fees.** Core `pay_invoice` has no fee limit. When the wallet offers
  [NWC-321](https://github.com/nostr-wallet-connect/nwc/blob/main/321.md)
  `pay`, every payment goes through it with `max_fee` set to the host's fee
  headroom, and `FEE_LIMIT_EXCEEDED` is reported as not submitted. NWC-321 lets
  a wallet ignore `max_fee`, so a payment may still cost more than that
  headroom. The fee the wallet reports is recorded as spend, so it counts
  against your daily budget either way; the connection's budget in the wallet
  is a second bound.
- **Amount-less invoices.** `make_invoice` requires an amount. When the wallet
  offers NWC-321 `receive`, amount-less invoices (`mesh-llm wallet fund-wallet`
  without `--amount-sats`) come from it instead; `receive` takes no expiry, so
  an invoice that would outlive the requested one is refused. Without
  `receive`, `fund-wallet` needs `--amount-sats`.
- **Arrival.** Notifications only report settlement, so a provider opens its
  output gate on settlement rather than on the earlier claiming signal.

The plugin opens one relay subscription before sending any request, so a
response or notification cannot be missed. Requests carry an expiration tag so
a late-delivered request is not executed after the plugin stopped waiting.
Settlement waits wake on notifications and poll `lookup_invoice` as the
backstop, since relays may drop notifications. A payment is reported as not
submitted only when the request never reached a relay or the wallet refused it
with a pre-routing error (`QUOTA_EXCEEDED`, `INSUFFICIENT_BALANCE`, ...);
everything else is uncertain and reconciled by payment hash.

The wallet identity pinned by the host is the wallet's node key when
`get_info` reports one, so replacing the connection to the same wallet keeps
the pin; otherwise it is the connection's key, and a new connection needs
`mesh-llm wallet unpin` first.
